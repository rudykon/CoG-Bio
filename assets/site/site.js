"use strict";
const results = [
  {
    "name": "MRP",
    "values": {
      "A4C": 28.86,
      "AOP": 22.33,
      "FA": 26.12,
      "FUGC": 14.74,
      "HC": 20.0,
      "IVC": 24.86,
      "PLAX": 23.3,
      "PSAX": 42.72,
      "Femur": 14.89,
      "Macro": 24.2
    }
  },
  {
    "name": "IAP",
    "values": {
      "A4C": 26.25,
      "AOP": 25.96,
      "FA": 25.92,
      "FUGC": 15.02,
      "HC": 19.66,
      "IVC": 34.63,
      "PLAX": 21.14,
      "PSAX": 46.23,
      "Femur": 13.33,
      "Macro": 25.35
    }
  },
  {
    "name": "Equal mean",
    "values": {
      "A4C": 26.18,
      "AOP": 23.52,
      "FA": 25.37,
      "FUGC": 14.54,
      "HC": 19.56,
      "IVC": 28.77,
      "PLAX": 21.29,
      "PSAX": 43.56,
      "Femur": 13.88,
      "Macro": 24.07
    }
  },
  {
    "name": "Consistency fusion",
    "values": {
      "A4C": 26.05,
      "AOP": 23.63,
      "FA": 25.34,
      "FUGC": 14.53,
      "HC": 19.55,
      "IVC": 27.4,
      "PLAX": 21.26,
      "PSAX": 43.87,
      "Femur": 13.46,
      "Macro": 23.9
    }
  },
  {
    "name": "CoG-Bio",
    "values": {
      "A4C": 26.05,
      "AOP": 23.63,
      "FA": 23.58,
      "FUGC": 14.53,
      "HC": 19.22,
      "IVC": 27.4,
      "PLAX": 21.26,
      "PSAX": 43.87,
      "Femur": 13.46,
      "Macro": 23.67
    }
  }
];
const chart = document.getElementById("result-chart");
const select = document.getElementById("task-select");
function renderChart(task) {
  const maximum = Math.ceil(Math.max(...results.map(row => row.values[task])) / 5) * 5;
  chart.replaceChildren();
  const description = [];
  results.forEach(row => {
    const value = row.values[task];
    const item = document.createElement("div");
    item.className = "chart-row" + (row.name === "CoG-Bio" ? " chart-final" : "");
    const label = document.createElement("span"); label.className = "chart-label"; label.textContent = row.name;
    const track = document.createElement("div"); track.className = "bar-track";
    const bar = document.createElement("span"); bar.className = "bar"; bar.style.width = `${value / maximum * 100}%`;
    track.append(bar);
    const score = document.createElement("span"); score.className = "chart-score"; score.textContent = value.toFixed(2);
    item.append(label, track, score); chart.append(item);
    description.push(`${row.name}: ${value.toFixed(2)} pixels`);
  });
  const axis = document.createElement("div"); axis.className = "chart-axis";
  [0, maximum / 2, maximum].forEach(value => { const tick = document.createElement("span"); tick.textContent = String(value); axis.append(tick); });
  chart.append(axis);
  chart.setAttribute("aria-label", `${task} MRE, lower is better. ${description.join("; ")}. The scale starts at zero.`);
  document.getElementById("chart-note").textContent = task === "Macro" ? "Task-macro MRE across all nine tasks, in original pixels." : `${task} landmark MRE, in original pixels. Lower is better.`;
}
select.addEventListener("change", () => renderChart(select.value));
renderChart(select.value);
const dialog = document.getElementById("figure-dialog");
const dialogImage = document.getElementById("dialog-image");
let lastFigure = null;
document.querySelectorAll("[data-zoom]").forEach(link => link.addEventListener("click", event => {
  if (event.ctrlKey || event.metaKey || event.shiftKey || !dialog.showModal) return;
  event.preventDefault();
  lastFigure = link;
  const image = link.querySelector("img");
  dialogImage.src = image.src; dialogImage.alt = image.alt;
  document.getElementById("dialog-title").textContent = link.closest("figure").querySelector("figcaption strong").textContent;
  document.getElementById("dialog-caption").textContent = link.closest("figure").querySelector("figcaption").textContent;
  document.getElementById("figure-original").href = link.href;
  dialog.showModal(); document.body.classList.add("dialog-open");
}));
document.getElementById("close-figure").addEventListener("click", () => dialog.close());
dialog.addEventListener("click", event => { if (event.target === dialog) { const r = dialog.getBoundingClientRect(); if (event.clientX < r.left || event.clientX > r.right || event.clientY < r.top || event.clientY > r.bottom) dialog.close(); } });
dialog.addEventListener("close", () => { document.body.classList.remove("dialog-open"); if (lastFigure) lastFigure.focus({preventScroll:true}); });
const copy = document.getElementById("copy-command");
copy.addEventListener("click", async () => {
  try { await navigator.clipboard.writeText("git clone https://github.com/rudykon/CoG-Bio.git"); copy.textContent = "Copied"; document.getElementById("copy-status").textContent = "Clone command copied."; }
  catch { document.getElementById("copy-status").textContent = "Select the command above to copy it."; }
});
