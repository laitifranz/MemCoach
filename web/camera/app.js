import { triggerHaptic } from "./haptics.js";

const videoEl = document.getElementById("camera-feed");
const canvasEl = document.getElementById("capture-canvas");
const scoreBtn = document.getElementById("score-btn");
const scoreFeedbackBtn = document.getElementById("score-feedback-btn");
const overlay = document.getElementById("loading-overlay");
const healthChip = document.getElementById("health-chip");
const scoreValue = document.getElementById("score-value");
const latencyLabel = document.getElementById("latency-label");
const feedbackChip = document.getElementById("feedback-chip");
const toastEl = document.getElementById("toast");
const scoreBadgeValue = document.getElementById("score-badge-value");
const feedbackDialog = document.getElementById("feedback-dialog");
const dialogText = document.getElementById("dialog-text");
const dialogClose = document.getElementById("dialog-close");
const resultsList = document.getElementById("results-list");
const resultsDelta = document.getElementById("results-delta");
const badgeDelta = document.getElementById("badge-delta");
const FEEDBACK_PLACEHOLDER =
  "Feedback will appear here after you run a score with feedback.";
const MAX_HISTORY = 10;
function deriveApiBase() {
  const override = new URLSearchParams(window.location.search).get("api");
  if (override) {
    return override.replace(/\/$/, "");
  }

  const url = new URL(window.location.href);
  if (!url.port) {
    if (url.protocol === "https:") {
      console.warn(
        "UI served over HTTPS without ?api override; falling back to same origin."
      );
      return "";
    }
    url.port = "8000";
  } else if (url.port !== "8000") {
    url.port = "8000";
  }

  return url.origin;
}

const API_BASE = deriveApiBase();
const HEALTH_ENDPOINT = `${API_BASE}/health`;
const SCORE_ENDPOINT = `${API_BASE}/score`;
const SCORE_FEEDBACK_ENDPOINT = `${API_BASE}/score-feedback`;

let mediaStream;
let lastFeedback = "";
const resultsHistory = [];

async function initCamera() {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: "environment" } },
      audio: false,
    });
    videoEl.srcObject = mediaStream;
  } catch (err) {
    showToast(`Camera error: ${err.message}`, true);
    triggerHaptic("error");
  }
}

async function healthCheck() {
  updateHealthChip("pending", "Checking server…");
  try {
    const res = await fetch(HEALTH_ENDPOINT, { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    if (data.status === "ok") {
      updateHealthChip("ok", "Server online");
      enableControls(true);
    } else {
      throw new Error("Unexpected response");
    }
  } catch (err) {
    updateHealthChip("error", "Server offline");
    enableControls(false);
    showToast(`Health check failed: ${err.message}`, true);
    triggerHaptic("error");
  }
}

function updateHealthChip(state, label) {
  healthChip.className = "chip";
  if (state === "ok") healthChip.classList.add("chip--ok");
  if (state === "error") healthChip.classList.add("chip--error");
  if (state === "pending") healthChip.classList.add("chip--pending");
  healthChip.textContent = label;
}

function enableControls(enabled) {
  scoreBtn.disabled = !enabled;
  scoreFeedbackBtn.disabled = !enabled;
}

function showOverlay(show) {
  overlay.classList.toggle("hidden", !show);
}

function showToast(message, isError = false) {
  toastEl.textContent = message;
  toastEl.classList.toggle("error", isError);
  toastEl.classList.remove("hidden");
  setTimeout(() => toastEl.classList.add("hidden"), 4000);
}

function captureBlob() {
  return new Promise((resolve, reject) => {
    if (!videoEl.videoWidth || !videoEl.videoHeight) {
      reject(new Error("Video stream not ready"));
      return;
    }
    const maxEdge = 1024;
    const { width, height } = scaleDimensions(videoEl.videoWidth, videoEl.videoHeight, maxEdge);
    canvasEl.width = width;
    canvasEl.height = height;
    const ctx = canvasEl.getContext("2d");
    ctx.drawImage(videoEl, 0, 0, width, height);
    canvasEl.toBlob(
      (blob) => {
        if (!blob) {
          reject(new Error("Failed to capture image"));
        } else {
          resolve(blob);
        }
      },
      "image/jpeg",
      0.8
    );
  });
}

function scaleDimensions(srcW, srcH, maxEdge) {
  const ratio = Math.min(1, maxEdge / Math.max(srcW, srcH));
  return { width: Math.round(srcW * ratio), height: Math.round(srcH * ratio) };
}

async function sendRequest(endpoint, mode = "score") {
  try {
    showOverlay(true);
    enableControls(false);
    const blob = await captureBlob();
    const form = new FormData();
    form.append("image", blob, "capture.jpg");

    const start = performance.now();
    const res = await fetch(endpoint, { method: "POST", body: form });
    const elapsed = performance.now() - start;

    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }

    const data = await res.json();
    updateResult(data, elapsed, mode);
    showOverlay(false);
    enableControls(true);
  } catch (err) {
    showOverlay(false);
    enableControls(true);
    showToast(err.message || "Request failed", true);
    triggerHaptic("error");
  }
}

function setScoreBadge(text) {
  scoreBadgeValue.textContent = text ?? "—";
}

function updateResult(payload, measuredLatency, mode = "score") {
  const score = Number(payload.score);
  if (!Number.isFinite(score)) {
    scoreValue.textContent = "Invalid score";
    setScoreBadge();
    return;
  }
  const formattedScore = score.toFixed(3);
  scoreValue.textContent = formattedScore;
  setScoreBadge(formattedScore);
  const latency = payload.latency_ms ?? measuredLatency;
  latencyLabel.textContent = latency ? `${latency.toFixed(0)} ms` : "";

  setFeedbackPreview(payload.feedback);
  triggerHaptic("success");
  recordResult({
    score,
    formattedScore,
    latency,
    timestamp: new Date(),
    mode,
  });
}

function setFeedbackPreview(feedbackText) {
  lastFeedback = (feedbackText || "").trim();
  if (lastFeedback) {
    const trimmed = lastFeedback.length > 200 ? `${lastFeedback.slice(0, 197)}…` : lastFeedback;
    feedbackChip.textContent = trimmed;
    feedbackChip.classList.remove("placeholder");
    feedbackChip.setAttribute("aria-disabled", "false");
  } else {
    feedbackChip.textContent = FEEDBACK_PLACEHOLDER;
    feedbackChip.classList.add("placeholder");
    feedbackChip.setAttribute("aria-disabled", "true");
  }
}

function recordResult(entry) {
  resultsHistory.unshift(entry);
  if (resultsHistory.length > MAX_HISTORY) {
    resultsHistory.pop();
  }
  renderResultsHistory();
  updateCurrentDelta();
}

function renderResultsHistory() {
  if (!resultsList) return;
  resultsList.innerHTML = "";
  if (!resultsHistory.length) {
    const placeholder = document.createElement("li");
    placeholder.className = "results-empty";
    placeholder.textContent = "Run a score to populate history.";
    resultsList.appendChild(placeholder);
    return;
  }

  resultsHistory.forEach((entry, index) => {
    const listItem = document.createElement("li");
    listItem.className = "result-item";
    if (index === 0) listItem.classList.add("result-item--latest");
    listItem.innerHTML = buildResultItemTemplate(entry, resultsHistory[index + 1]);
    resultsList.appendChild(listItem);
  });
}

function buildResultItemTemplate(entry, previousEntry) {
  const deltaInfo = computeDelta(entry.score, previousEntry?.score);
  const modeLabel = entry.mode === "score-feedback" ? "Score + Feedback" : "Score Only";
  const modeClass = entry.mode === "score-feedback" ? "mode-chip mode-chip--feedback" : "mode-chip";

  return `
    <div class="result-item__primary">
      <span class="result-item__score">${entry.formattedScore}</span>
      <span class="result-item__delta ${deltaInfo.variant}">${deltaInfo.text}</span>
    </div>
    <div class="result-item__meta">
      <span class="${modeClass}">${modeLabel}</span>
      <span class="meta">${formatTimestamp(entry.timestamp)}</span>
    </div>
  `;
}

function computeDelta(currentScore, previousScore) {
  if (!Number.isFinite(previousScore)) {
    return { text: "—", variant: "neutral" };
  }

  const diff = currentScore - previousScore;
  const absDiff = Math.abs(diff);
  if (absDiff < 0.0005) {
    return { text: "0.000 vs prev", variant: "neutral" };
  }
  const prefix = diff > 0 ? "+" : "";
  const variant = diff > 0 ? "positive" : "negative";
  return { text: `${prefix}${diff.toFixed(3)} vs prev`, variant };
}

function formatTimestamp(date) {
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function updateCurrentDelta() {
  const hasComparison = resultsHistory.length >= 2;
  if (!hasComparison) {
    setDeltaIndicator(resultsDelta, "—", "delta-pill--neutral");
    setDeltaIndicator(badgeDelta, "—", "delta-pill--neutral", "delta-pill--compact");
    return;
  }

  const diff = resultsHistory[0].score - resultsHistory[1].score;
  const absDiff = Math.abs(diff);
  let variant = "delta-pill--neutral";
  if (absDiff >= 0.0005) {
    variant = diff > 0 ? "delta-pill--positive" : "delta-pill--negative";
  }
  const prefix = diff > 0 ? "+" : "";
  const valueText = absDiff < 0.0005 ? "0.000" : `${prefix}${diff.toFixed(3)}`;
  setDeltaIndicator(resultsDelta, valueText, variant);
  const percentInfo = computePercentDelta(diff, resultsHistory[1].score);
  setDeltaIndicator(badgeDelta, percentInfo.text, percentInfo.variant, "delta-pill--compact");
}

function setDeltaIndicator(element, valueText, variant, extraClasses = "") {
  if (!element) return;
  const classNames = ["delta-pill", extraClasses, variant].filter(Boolean).join(" ");
  element.textContent = `Δ ${valueText}`;
  element.className = classNames;
}

function computePercentDelta(diff, baselineScore) {
  if (!Number.isFinite(baselineScore) || Math.abs(baselineScore) < 1e-6) {
    return { text: "—", variant: "delta-pill--neutral" };
  }
  const percent = (diff / baselineScore) * 100;
  const absPercent = Math.abs(percent);
  let variant = "delta-pill--neutral";
  if (absPercent >= 0.05) {
    variant = percent > 0 ? "delta-pill--positive" : "delta-pill--negative";
  }
  const prefix = percent > 0 ? "+" : "";
  const text = absPercent < 0.05 ? "0.0%" : `${prefix}${percent.toFixed(1)}%`;
  return { text, variant };
}

scoreBtn.addEventListener("click", () => {
  triggerHaptic("selection");
  sendRequest(SCORE_ENDPOINT, "score");
});
scoreFeedbackBtn.addEventListener("click", () => {
  triggerHaptic("selection");
  sendRequest(SCORE_FEEDBACK_ENDPOINT, "score-feedback");
});
feedbackChip.addEventListener("click", () => {
  if (!lastFeedback) return;
  dialogText.textContent = lastFeedback;
  feedbackDialog.showModal();
  triggerHaptic("selection");
});
feedbackChip.addEventListener("keydown", (evt) => {
  if (!lastFeedback) return;
  if (evt.key === "Enter" || evt.key === " ") {
    evt.preventDefault();
    dialogText.textContent = lastFeedback;
    feedbackDialog.showModal();
    triggerHaptic("selection");
  }
});
dialogClose.addEventListener("click", () => feedbackDialog.close());

document.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    enableControls(false);
    showOverlay(false);
  } else if (healthChip.classList.contains("chip--ok")) {
    enableControls(true);
  }
});

(async function bootstrap() {
  await healthCheck();
  setInterval(healthCheck, 30000);

  if (!navigator.mediaDevices?.getUserMedia) {
    showToast("Camera API not supported", true);
    triggerHaptic("error");
    return;
  }
  await initCamera();
  setFeedbackPreview();
  updateCurrentDelta();
})();
