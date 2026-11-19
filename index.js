// Runs the two PyTorch -> ONNX models in the browser with onnxruntime-web.
// Both models take input "input1" with shape [1, 8] (raw, un-normalized values)
// and return "output1".

const sessions = {};

async function getSession(file) {
  if (!sessions[file]) {
    sessions[file] = await ort.InferenceSession.create("./" + file);
  }
  return sessions[file];
}

function readInputs(prefix) {
  const x = new Float32Array(8);
  for (let i = 0; i < 8; i++) {
    x[i] = parseFloat(document.getElementById(prefix + i).value) || 0;
  }
  return new ort.Tensor("float32", x, [1, 8]);
}

function setInputs(prefix, values) {
  values.forEach((v, i) => { document.getElementById(prefix + i).value = v; });
}

async function runWins() {
  const out = document.getElementById("predWins");
  try {
    const session = await getSession("wins_regression.onnx");
    const results = await session.run({ input1: readInputs("c") });
    const wins = results.output1.data[0];
    out.innerHTML = `Predicted wins: <b>${wins.toFixed(1)}</b> (record ≈ ${Math.round(wins)}-${162 - Math.round(wins)})`;
  } catch (e) {
    console.error("ONNX runtime error:", e);
    out.textContent = "Error: " + e.message;
  }
}

async function runHOF() {
  const out = document.getElementById("predHOF");
  try {
    const session = await getSession("hof_classifier.onnx");
    const results = await session.run({ input1: readInputs("d") });
    const p = results.output1.data; // [P(not HOF), P(HOF)]
    const label = p[1] >= 0.5 ? "Hall of Famer" : "Not a Hall of Famer";
    out.innerHTML = `Prediction: <b>${label}</b><br>
      P(HOF) = ${(p[1] * 100).toFixed(1)}%
      <div class="bar"><div style="width:${(p[1] * 100).toFixed(1)}%"></div></div>`;
  } catch (e) {
    console.error("ONNX runtime error:", e);
    out.textContent = "Error: " + e.message;
  }
}

function presetTeam(name) {
  // R, RA, H, HR, BB, SB, SOA, E
  if (name === "cubs")     setInputs("c", [808, 556, 1409, 199, 656, 66, 1441, 101]);
  if (name === "mariners") setInputs("c", [927, 627, 1637, 169, 614, 174, 1051, 83]);
  if (name === "tigers")   setInputs("c", [591, 928, 1312, 153, 443, 98, 764, 138]);
  runWins();
}

function presetHOF(name) {
  // career stats: G, R, H, HR, RBI, SB, BB, AVG
  if (name === "gwynn")   setInputs("d", [2440, 1383, 3141, 135, 1138, 319, 790, 0.338]);
  if (name === "griffey") setInputs("d", [2671, 1662, 2781, 630, 1836, 184, 1312, 0.284]);
  if (name === "garvey")  setInputs("d", [2332, 1143, 2599, 272, 1308, 83, 479, 0.294]);
  runHOF();
}
