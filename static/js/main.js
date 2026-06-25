const textarea  = document.getElementById("articleText");
const charCount = document.getElementById("charCount");

if (textarea) {
  textarea.addEventListener("input", () => {
    const len = textarea.value.length;
    charCount.textContent = `${len.toLocaleString()} character${len !== 1 ? "s" : ""}`;
  });
}

async function analyzeArticle() {
  hideCards();

  const text = textarea.value.trim();
  if (!text || text.length < 20) {
    showError("Please enter a news article (at least 20 characters).");
    return;
  }

  showLoading(true);

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text })
    });

    const data = await response.json();

    if (!response.ok || data.error) {
      showError(data.error || "Server error — please try again.");
      return;
    }

    renderResult(data);

  } catch (err) {
    showError("Could not reach the server. Make sure app.py is running on port 5000.");
  } finally {
    showLoading(false);
  }
}

function renderResult(data) {
  const { label, confidence, highlighted_text, keywords, fake_prob, real_prob } = data;
  const isFake = label === "FAKE";

  const chip = document.getElementById("verdictChip");
  chip.className = "verdict-chip " + (isFake ? "fake" : "real");
  document.getElementById("verdictLabel").textContent = label;

  const banner = document.getElementById("verdictBanner");
  banner.className = "verdict-banner " + (isFake ? "is-fake" : "is-real");

  document.getElementById("verdictSummary").textContent =
    `${confidence}% confidence — ${isFake
      ? "this article shows signs of misinformation."
      : "this article appears to be credible."}`;

  const circumference = 314;
  const ringFill = document.getElementById("ringFill");
  ringFill.style.strokeDashoffset = circumference - (confidence / 100) * circumference;
  ringFill.style.stroke = isFake ? "#ff4d6d" : "#00e5a0";
  ringFill.style.filter = isFake
    ? "drop-shadow(0 0 6px rgba(255,77,109,0.6))"
    : "drop-shadow(0 0 6px rgba(0,229,160,0.6))";
  document.getElementById("ringText").textContent = confidence + "%";
  document.getElementById("ringText").style.color = isFake ? "#ff4d6d" : "#00e5a0";

  setBar("fakeBar", "fakePct", fake_prob);
  setBar("realBar", "realPct", real_prob);

  const chipsEl = document.getElementById("keywordChips");
  chipsEl.innerHTML = keywords.map(kw => `<span class="chip">${kw}</span>`).join("");

  document.getElementById("highlightedText").innerHTML = highlighted_text;

  const resultCard = document.getElementById("resultCard");
  resultCard.style.display = "block";
  document.getElementById("errorCard").classList.remove("visible");
  resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function setBar(barId, pctId, value) {
  document.getElementById(barId).style.width = value + "%";
  document.getElementById(pctId).textContent = value + "%";
}

async function loadMetrics() {
  try {
    const res  = await fetch("/metrics");
    const data = await res.json();

    if (data.error) {
      document.getElementById("cmDisplay").innerHTML =
        `<p style="color:#fbbf24;font-size:13px;padding:1rem;">${data.error}</p>`;
      return;
    }

    renderConfusionMatrix(data);
    renderStats(data);
    document.getElementById("metrics").scrollIntoView({ behavior: "smooth" });

  } catch (e) {
    document.getElementById("cmDisplay").innerHTML =
      `<p style="color:#ff4d6d;font-size:13px;padding:1rem;">Failed to load metrics.</p>`;
  }
}

function renderConfusionMatrix(data) {
  const cm = data.confusion_matrix;

  const trueFakePredFake = cm[0][0];
  const trueFakePredReal = cm[0][1];
  const trueRealPredFake = cm[1][0];
  const trueRealPredReal = cm[1][1];

  document.getElementById("cmDisplay").innerHTML = `
    <div style="padding:1.5rem;width:100%;">
      <div class="cm-table">
        <div class="cm-cell cm-header"></div>
        <div class="cm-cell cm-header">Pred FAKE</div>
        <div class="cm-cell cm-header">Pred REAL</div>

        <div class="cm-cell cm-label">True FAKE</div>
        <div class="cm-cell cm-tn">${trueFakePredFake.toLocaleString()}</div>
        <div class="cm-cell cm-fp">${trueFakePredReal.toLocaleString()}</div>

        <div class="cm-cell cm-label">True REAL</div>
        <div class="cm-cell cm-fn">${trueRealPredFake.toLocaleString()}</div>
        <div class="cm-cell cm-tp">${trueRealPredReal.toLocaleString()}</div>
      </div>

      <p class="cm-legend">Green = correct · Red = errors</p>
    </div>
  `;
}

function renderStats(data) {
  const r = data.classification_report;
  const rows = [
    ["Overall accuracy",  data.accuracy + "%"],
    ["FAKE precision",    pct(r["FAKE"]?.precision)],
    ["FAKE recall",       pct(r["FAKE"]?.recall)],
    ["REAL precision",    pct(r["REAL"]?.precision)],
    ["REAL recall",       pct(r["REAL"]?.recall)],
    ["Macro F1",          pct(r["macro avg"]?.["f1-score"])],
  ];

  document.getElementById("statsDisplay").innerHTML =
    `<div style="padding:1.25rem 1.5rem;width:100%;">` +
    rows.map(([name, val]) => `
      <div class="stat-item">
        <span class="stat-name">${name}</span>
        <span class="stat-value">${val}</span>
      </div>`).join("") +
    `</div>`;
}

function pct(val) {
  if (val == null) return "-";
  return (val * 100).toFixed(1) + "%";
}

function showLoading(visible) {
  document.getElementById("loadingOverlay").style.display = visible ? "flex" : "none";
}

function showError(msg) {
  document.getElementById("resultCard").style.display = "none";
  const ec = document.getElementById("errorCard");
  document.getElementById("errorMsg").textContent = msg;
  ec.classList.add("visible");
  ec.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function hideCards() {
  document.getElementById("resultCard").style.display = "none";
  document.getElementById("errorCard").classList.remove("visible");
}

document.addEventListener("keydown", e => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") analyzeArticle();
});