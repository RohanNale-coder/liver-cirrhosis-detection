const express = require('express');
const fs = require('fs');
const path = require('path');
const { RandomForestClassifier } = require('ml-random-forest');

const app = express();
app.use(express.json());
app.use(express.static(path.join(__dirname)));

// Load trained model
const model = RandomForestClassifier.load(
  JSON.parse(fs.readFileSync('./model.json', 'utf8'))
);

// Encoding helpers
const encode = {
  status: v => ({ C: 0, CL: 1, D: 2 }[v]),
  drug: v => ({ Placebo: 0, "D-penicillamine": 1 }[v]),
  sex: v => ({ F: 0, M: 1 }[v]),
  binary: v => (v === 'Y' ? 1 : 0),
  edema: v => ({ N: 0, S: 1, Y: 2 }[v])
};

app.post('/predict', (req, res) => {
  const d = req.body;

  const input = [[
    +d.n_days,
    encode.status(d.status),
    encode.drug(d.drug),
    +d.age,
    encode.sex(d.sex),
    encode.binary(d.ascites),
    encode.binary(d.hepatomegaly),
    encode.binary(d.spiders),
    encode.edema(d.edema),
    +d.bilirubin,
    +d.cholesterol,
    +d.albumin,
    +d.copper,
    +d.alk_phos,
    +d.sgot,
    +d.triglycerides,
    +d.platelets,
    +d.prothrombin
  ]];

  const stage = model.predict(input)[0];

  res.json({
    predicted_stage: stage,
    meaning:
      stage === 1 ? 'Early Stage' :
      stage === 2 ? 'Intermediate Stage' :
      'Advanced Stage'
  });
});

app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'index.html'));
});

app.listen(3000, () =>
  console.log('🚀 Server running at http://localhost:3000')
);
