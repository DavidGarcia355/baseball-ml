// Runs the two PyTorch -> ONNX models in the browser with onnxruntime-web.
// Both models take input "input1" with shape [1, 8] (raw stats) and return "output1".

const sessions = {};
let TEAMS = [], PLAYERS = [];
let selectedTeam = null, selectedPlayer = null;

async function getSession(file) {
  if (!sessions[file]) sessions[file] = await ort.InferenceSession.create("./" + file);
  return sessions[file];
}

function readValues(prefix) {
  const x = [];
  for (let i = 0; i < 8; i++) x.push(parseFloat(document.getElementById(prefix + i).value) || 0);
  return x;
}

function setInputs(prefix, values) {
  values.forEach((v, i) => { document.getElementById(prefix + i).value = v; });
}

// ---------------- Team wins (regression) ----------------
async function runWins() {
  const out = document.getElementById("predWins");
  try {
    const session = await getSession("wins_regression.onnx");
    // The model was trained on 162-game seasons. For other season lengths,
    // scale the counting stats to 162 games, predict, then scale wins back.
    const games = selectedTeam ? selectedTeam.g : 162;
    const k = 162 / games;
    const x = readValues("c").map(v => v * k);
    const results = await session.run({ input1: new ort.Tensor("float32", Float32Array.from(x), [1, 8]) });
    const wins = results.output1.data[0] / k;
    let html = `Predicted wins: <b>${wins.toFixed(1)}</b> (record ≈ ${Math.round(wins)}-${games - Math.round(wins)})`;
    if (selectedTeam) {
      const diff = selectedTeam.w - wins;
      const verdict = Math.abs(diff) < 2 ? "right about what their stats predicted"
        : diff > 0 ? `<b>${diff.toFixed(1)} more</b> than their stats predicted (lucky)`
                   : `<b>${(-diff).toFixed(1)} fewer</b> than their stats predicted (unlucky)`;
      html += `<br>Actual: <b>${selectedTeam.w}-${selectedTeam.l}</b> in ${games} games. They won ${verdict}.`;
      if (selectedTeam.y < 1962) html += `<br><span class="muted">Note: pre-1962 is outside the training data, so this is a rougher estimate.</span>`;
    }
    out.innerHTML = html;
  } catch (e) {
    console.error("ONNX runtime error:", e);
    out.textContent = "Error: " + e.message;
  }
}

// ---------------- Hall of Fame (classification) ----------------
const STATUS = {
  hof: "Actual: <b>Hall of Famer</b>",
  no: "Actual: <b>Not in the Hall of Fame</b>",
  banned: "Actual: <b>Banned from the Hall of Fame</b>",
  short: "Actual: <b>Not eligible</b> (fewer than 10 seasons)",
  recent: "Retired after 2003 or still active in 2021, so this player wasn't in the training data. This is just the model's guess.",
};

async function runHOF() {
  const out = document.getElementById("predHOF");
  try {
    const session = await getSession("hof_classifier.onnx");
    const x = Float32Array.from(readValues("d"));
    const results = await session.run({ input1: new ort.Tensor("float32", x, [1, 8]) });
    const p = results.output1.data; // [P(not HOF), P(HOF)]
    const label = p[1] >= 0.5 ? "Hall of Famer" : "Not a Hall of Famer";
    let html = `Prediction: <b>${label}</b><br>P(HOF) = ${(p[1] * 100).toFixed(1)}%
      <div class="bar"><div style="width:${(p[1] * 100).toFixed(1)}%"></div></div>`;
    if (selectedPlayer) html += `<span class="muted">${STATUS[selectedPlayer.st]}</span>`;
    out.innerHTML = html;
  } catch (e) {
    console.error("ONNX runtime error:", e);
    out.textContent = "Error: " + e.message;
  }
}

// ---------------- Search boxes ----------------
function setupSearch(inputId, listId, getItems, label, onPick) {
  const input = document.getElementById(inputId);
  const list = document.getElementById(listId);
  input.addEventListener("input", () => {
    const words = input.value.toLowerCase().split(/\s+/).filter(Boolean);
    list.innerHTML = "";
    if (!words.length) return;
    const matches = getItems().filter(it => {
      const text = label(it).toLowerCase();
      return words.every(w => text.includes(w));
    }).slice(0, 10);
    matches.forEach(it => {
      const li = document.createElement("li");
      li.textContent = label(it);
      li.addEventListener("mousedown", () => { input.value = label(it); list.innerHTML = ""; onPick(it); });
      list.appendChild(li);
    });
  });
  input.addEventListener("blur", () => setTimeout(() => { list.innerHTML = ""; }, 150));
}

const teamLabel = t => `${t.y} ${t.n} (${t.w}-${t.l})`;
const playerLabel = p => `${p.n} (${p.y})`;

// If the user edits a stat by hand, it's no longer that exact team/player,
// but we keep the selection so the actual record / status still shows.
["c", "d"].forEach(prefix => {
  for (let i = 0; i < 8; i++) {
    document.getElementById(prefix + i).addEventListener("change", () => prefix === "c" ? runWins() : runHOF());
  }
});

setupSearch("teamSearch", "teamResults", () => TEAMS, teamLabel, t => {
  selectedTeam = t; setInputs("c", t.s); runWins();
});
setupSearch("playerSearch", "playerResults", () => PLAYERS, playerLabel, p => {
  selectedPlayer = p; setInputs("d", p.s); runHOF();
});

// Load the data, then show a default team and player.
Promise.all([fetch("teams.json").then(r => r.json()), fetch("players.json").then(r => r.json())])
  .then(([teams, players]) => {
    TEAMS = teams; PLAYERS = players;
    selectedTeam = TEAMS.find(t => t.y === 2016 && t.n === "Chicago Cubs");
    document.getElementById("teamSearch").value = teamLabel(selectedTeam);
    setInputs("c", selectedTeam.s); runWins();
    selectedPlayer = PLAYERS.find(p => p.n === "Tony Gwynn");
    document.getElementById("playerSearch").value = playerLabel(selectedPlayer);
    setInputs("d", selectedPlayer.s); runHOF();
  });
