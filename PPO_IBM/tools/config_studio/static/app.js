let FIELDS = [];
let VALUES = {};

async function loadFields() {
  const res = await fetch("/api/fields");
  const data = await res.json();
  FIELDS = data.fields;
  VALUES = data.values;
  renderGroup("gate-table", ["D0 gate", "D1 gate", "D2 gate"]);
  renderGroup("demotion-table", ["Demotion"]);
  renderGroup("expert-table", ["Expert law"]);
  document.getElementById("log-curriculum").textContent = (data.curriculum_log || []).join("\n") || "(no commits yet)";
  document.getElementById("log-expert").textContent = (data.expert_log || []).join("\n") || "(no commits yet)";
}

function renderGroup(tableId, groups) {
  const table = document.getElementById(tableId);
  table.innerHTML = "<tr><th>Field</th><th>Value</th><th></th><th>Status</th></tr>";
  for (const group of groups) {
    const fields = FIELDS.filter(f => f.group === group);
    if (!fields.length) continue;
    const header = document.createElement("tr");
    header.innerHTML = `<td colspan="4" style="padding-top:0.8rem;color:#888;font-weight:600">${group}</td>`;
    table.appendChild(header);
    for (const f of fields) {
      const tr = document.createElement("tr");
      tr.id = `row-${f.id}`;
      const val = VALUES[f.id];
      tr.innerHTML = `
        <td>${f.label}</td>
        <td><input type="number" step="any" id="input-${f.id}" value="${val === null ? "" : val}"></td>
        <td><button data-id="${f.id}">Save</button></td>
        <td class="row-saved" id="status-${f.id}">${val === null ? "read error" : "saved"}</td>`;
      table.appendChild(tr);
      const input = tr.querySelector("input");
      const status = tr.querySelector(`#status-${f.id}`);
      input.addEventListener("input", () => {
        status.className = "row-dirty";
        status.textContent = "unsaved";
      });
      tr.querySelector("button").addEventListener("click", () => saveField(f.id));
    }
  }
}

async function saveField(id) {
  const input = document.getElementById(`input-${id}`);
  const status = document.getElementById(`status-${id}`);
  const value = parseFloat(input.value);
  status.className = "row-dirty";
  status.textContent = "saving...";
  try {
    const res = await fetch("/api/fields/update", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id, value }),
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    VALUES[id] = data.value;
    input.value = data.value;
    status.className = "row-saved";
    status.textContent = "saved (git committed)";
  } catch (e) {
    status.className = "row-error";
    status.textContent = `error: ${e.message}`;
  }
}

function currentValue(id) {
  const input = document.getElementById(`input-${id}`);
  return input ? parseFloat(input.value) : VALUES[id];
}

async function runPreview() {
  const status = document.getElementById("preview-status");
  const out = document.getElementById("preview-result");
  const difficulty = parseInt(document.getElementById("preview-difficulty").value, 10);
  const n_episodes = parseInt(document.getElementById("preview-n").value, 10);

  const gatePrefix = ["d0", "d1", "d2"][difficulty];
  const body = {
    difficulty, n_episodes,
    stir_min: currentValue("expert_stir_min"), stir_max: currentValue("expert_stir_max"),
    light_min: currentValue("expert_light_min"), light_max: currentValue("expert_light_max"),
    od_setpoint: currentValue("expert_od_setpoint"), gain: currentValue("expert_gain"),
    frac_cap: currentValue("expert_frac_cap"),
    gate_harvest: currentValue(`${gatePrefix}_harvest`), gate_p25: currentValue(`${gatePrefix}_p25`),
    gate_crash: currentValue(`${gatePrefix}_crash`), gate_od: currentValue(`${gatePrefix}_od`),
  };

  status.textContent = `running ${n_episodes} episodes at D${difficulty}...`;
  out.innerHTML = "";
  try {
    const res = await fetch("/api/preview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const r = await res.json();
    if (r.error) throw new Error(r.error);
    status.textContent = "";
    out.innerHTML = `
      <table>
        <tr><th>Metric</th><th>Result</th><th>Gate</th></tr>
        <tr><td>median harvested_mg</td><td>${r.median_harvested_mg.toFixed(1)}</td><td>&gt;= ${r.gate.harvest}</td></tr>
        <tr><td>p25 harvested_mg</td><td>${r.p25_harvested_mg.toFixed(1)}</td><td>&gt;= ${r.gate.p25}</td></tr>
        <tr><td>median time_avg_od</td><td>${r.median_time_avg_od.toFixed(4)}</td><td>&gt;= ${r.gate.od}</td></tr>
        <tr><td>crash rate</td><td>${(r.crash_rate * 100).toFixed(1)}%</td><td>&lt;= ${(r.gate.crash * 100).toFixed(0)}%</td></tr>
      </table>
      <div class="${r.passes_gate ? "pass" : "fail"}">${r.passes_gate ? "PASSES gate" : "FAILS gate"}</div>
      <div class="field-desc">stir=${r.stir.toFixed(1)} light=${r.light.toFixed(1)} (drawn once, held constant across the sweep)</div>`;
  } catch (e) {
    status.textContent = `error: ${e.message}`;
  }
}

document.getElementById("preview-run").addEventListener("click", runPreview);
loadFields();
