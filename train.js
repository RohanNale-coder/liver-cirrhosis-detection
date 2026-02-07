const fs = require('fs');
const path = require('path');
const csv = require('csv-parser');
const { fork } = require('child_process');
const { RandomForestClassifier } = require('ml-random-forest');

console.log('1️⃣ Script started');

const CSV_PATH = path.join(__dirname, 'data', 'pbc.csv');

function loadCSV() {
  return new Promise((resolve, reject) => {
    const X = [];
    const y = [];
    fs.createReadStream(CSV_PATH)
      .pipe(csv())
      .on('data', (row) => {
        // parse numbers and validate
        const b = Number(row.Bilirubin);
        const a = Number(row.Albumin);
        const c = Number(row.Copper);
        const p = Number(row.Platelets);
        const pr = Number(row.Prothrombin);
        const s = Number(row.Stage);
        if (!Number.isFinite(b) || !Number.isFinite(a) || !Number.isFinite(c) || !Number.isFinite(p) || !Number.isFinite(pr) || !Number.isFinite(s)) return;

        X.push([b, a, c, p, pr]);
        y.push(s);
      })
      .on('end', () => resolve({ X, y }))
      .on('error', reject);
  });
}

function msToHuman(ms) {
  if (ms <= 0) return '0s';
  const s = Math.round(ms / 1000);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  if (h) return `${h}h ${m}m ${sec}s`;
  if (m) return `${m}m ${sec}s`;
  return `${sec}s`;
}

async function controller() {
  try {
    console.log('2️⃣ Loading CSV (for benchmark)...');
    const { X, y } = await loadCSV();
    console.log(`📊 Loaded ${X.length} records`);

    const DESIRED_ESTIMATORS = Number(process.env.N_ESTIMATORS) || 15;
    const BENCH_ESTIMATORS = Math.max(1, Math.floor(DESIRED_ESTIMATORS / 5) || 2);

    console.log(`🔎 Running quick benchmark with ${BENCH_ESTIMATORS} estimators...`);
    const benchModel = new RandomForestClassifier({
      nEstimators: BENCH_ESTIMATORS,
      maxFeatures: 0.9,
      replacement: true,
      seed: 42,
      noOOB: true
    });

    const benchStart = Date.now();
    benchModel.train(X, y);
    const benchElapsed = Date.now() - benchStart;
    const estimatedTotal = Math.max(benchElapsed * (DESIRED_ESTIMATORS / BENCH_ESTIMATORS), 1);

    console.log(`⚡ Bench: ${benchElapsed} ms -> estimated total ${Math.round(estimatedTotal)} ms (${msToHuman(estimatedTotal)})`);

    console.log('3️⃣ Forking worker to run full training');
    const worker = fork(__filename, [], { env: { WORKER: '1', N_ESTIMATORS: String(DESIRED_ESTIMATORS) } });

    const start = Date.now();
    const interval = setInterval(() => {
      const elapsed = Date.now() - start;
      const remaining = Math.max(0, Math.round(estimatedTotal - elapsed));
      process.stdout.write(`\r⏳ Elapsed: ${msToHuman(elapsed)} — Est. remaining: ${msToHuman(remaining)} `);
    }, 1000);

    worker.on('message', (msg) => {
      if (msg && msg.status === 'done') {
        clearInterval(interval);
        const total = Date.now() - start;
        console.log(`\n✅ Worker finished training in ${msToHuman(total)} (reported ${msg.time} ms)`);
        process.exit(0);
      }
      if (msg && msg.status === 'error') {
        clearInterval(interval);
        console.error('\n❌ Worker error:', msg.error);
        process.exit(1);
      }
    });

    worker.on('exit', (code) => {
      clearInterval(interval);
      if (code !== 0) console.error(`\n❌ Worker exited with code ${code}`);
    });
  } catch (err) {
    console.error('❌ Controller error:', err.message || err);
    process.exit(1);
  }
}

async function worker() {
  try {
    console.log('🔧 Worker: loading CSV and training full model...');
    const { X, y } = await loadCSV();
    const N = Number(process.env.N_ESTIMATORS) || 50;
    const model = new RandomForestClassifier({
      nEstimators: N,
      maxFeatures: 0.9,
      replacement: true,
      seed: 42,
      noOOB: true
    });

    const start = Date.now();
    model.train(X, y);
    const elapsed = Date.now() - start;

    fs.writeFileSync(path.join(__dirname, 'model.json'), JSON.stringify(model.toJSON()));

    if (process.send) process.send({ status: 'done', time: elapsed });
    console.log(`✅ Worker: training complete in ${msToHuman(elapsed)}`);
    process.exit(0);
  } catch (err) {
    if (process.send) process.send({ status: 'error', error: err.message || String(err) });
    console.error('❌ Worker error:', err);
    process.exit(1);
  }
}

if (process.env.WORKER === '1') {
  worker();
} else {
  controller();
}
