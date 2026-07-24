import { triggerHaptic } from "./haptics.js";
import { renderOrb, clearOrb } from "./orb.js";

// ---------------------------------------------------------------------------
// API base URL resolution
// ---------------------------------------------------------------------------
function resolveApiBase() {
  const params = new URLSearchParams(window.location.search);
  const override = params.get("api");
  if (override) return override.replace(/\/$/, "");
  if (location.protocol === "https:") return "";
  return `http://${location.hostname}:8001`;
}

const API_BASE = resolveApiBase();
const HEALTH_ENDPOINT = `${API_BASE}/health`;
const EDIT_ENDPOINT = `${API_BASE}/edit`;
const LEADERBOARD_ENDPOINT = `${API_BASE}/leaderboard`;

// Stable session ID — persists across page refreshes, identifies this browser tab
function getSessionId() {
  let id = localStorage.getItem("memcoach_session");
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem("memcoach_session", id);
  }
  return id;
}
const SESSION_ID = getSessionId();

// ---------------------------------------------------------------------------
// DOM refs
// ---------------------------------------------------------------------------
const bodyEl          = document.body;
const videoEl         = document.getElementById("videoEl");
const canvasEl        = document.getElementById("canvasEl");
const captureBtn      = document.getElementById("captureBtn");
const frozenPreview   = document.getElementById("frozenPreview");
const frozenOrbSlots  = [document.getElementById("frozenOrbA"), document.getElementById("frozenOrbB")];
const frozenOrbLabel  = document.getElementById("frozenOrbLabel");
const STEP_ORB_STATES = ["solving", "composing", "shaping"];
const STEP_LABELS     = ["Analyzing memorability", "Generating feedback", "Editing with FLUX"];
const compareWrapper  = document.getElementById("compareWrapper");
const compareContainer= document.getElementById("compareContainer");
const compareDivider  = document.getElementById("compareDivider");
const afterSide       = document.getElementById("afterSide");
const imgBefore       = document.getElementById("imgBefore");
const imgAfter        = document.getElementById("imgAfter");
const scoreBefore     = document.getElementById("scoreBefore");
const scoreAfter      = document.getElementById("scoreAfter");
const improvementChip = document.getElementById("improvementChip");
const latencyLabel    = document.getElementById("latencyLabel");
const feedbackText    = document.getElementById("feedbackText");
const retryBtn        = document.getElementById("retryBtn");
const healthChip       = document.getElementById("healthChip");
const connectedCount   = document.getElementById("connectedCount");
const leaderboardBtn  = document.getElementById("leaderboardBtn");
const editCountBadge  = document.getElementById("editCountBadge");
const leaderboardBackdrop = document.getElementById("leaderboardBackdrop");
const closeLeaderboard    = document.getElementById("closeLeaderboard");
const leaderboardList     = document.getElementById("leaderboardList");
const leaderboardSubtitle = document.getElementById("leaderboardSubtitle");
const toastEl         = document.getElementById("toast");

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let mediaStream = null;
let serverOnline = false;
let capturedBlob = null;
let capturedDataUrl = null;
let lbPollTimer = null;

// ---------------------------------------------------------------------------
// Display mode (presenter screen)
// ---------------------------------------------------------------------------
const isDisplayMode = new URLSearchParams(window.location.search).get("mode") === "display";
if (isDisplayMode) {
  bodyEl.classList.add("display-mode");
  injectDisplayLeaderboard();
  startDisplayPoll();
}

function injectDisplayLeaderboard() {
  const section = document.createElement("section");
  section.className = "display-leaderboard";
  section.innerHTML = `
    <h2>MemCoach — Live Leaderboard</h2>
    <p class="display-lb-subtitle" id="displaySubtitle">Loading…</p>
    <ol class="display-lb-list leaderboard-list" id="displayList"></ol>
  `;
  bodyEl.appendChild(section);
}

async function startDisplayPoll() {
  await fetchAndRenderLeaderboard(
    document.getElementById("displayList"),
    document.getElementById("displaySubtitle"),
  );
  setInterval(() => {
    fetchAndRenderLeaderboard(
      document.getElementById("displayList"),
      document.getElementById("displaySubtitle"),
    );
  }, 10_000);
}

// ---------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------
async function initCamera() {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: "environment" } },
      audio: false,
    });
    videoEl.srcObject = mediaStream;
  } catch (err) {
    showToast("Camera access denied", true);
  }
}

// ---------------------------------------------------------------------------
// Health check
// ---------------------------------------------------------------------------
async function checkHealth() {
  try {
    const res = await fetch(`${HEALTH_ENDPOINT}?session=${SESSION_ID}`, { signal: AbortSignal.timeout(4000) });
    serverOnline = res.ok;
    if (res.ok) {
      const data = await res.json();
      const n = data.connected ?? 0;
      connectedCount.textContent = `${n} connected`;
      connectedCount.hidden = false;
    }
  } catch {
    serverOnline = false;
    connectedCount.hidden = true;
  }
  healthChip.textContent = "●";
  healthChip.className = `health-chip ${serverOnline ? "online" : "offline"}`;
  captureBtn.disabled = !serverOnline;
}

// ---------------------------------------------------------------------------
// Image capture + compression
// ---------------------------------------------------------------------------
async function captureFrame() {
  const MAX_EDGE = 1024;
  const vw = videoEl.videoWidth;
  const vh = videoEl.videoHeight;
  if (!vw || !vh) throw new Error("No video frame available");

  const scale = Math.min(1, MAX_EDGE / Math.max(vw, vh));
  canvasEl.width  = Math.round(vw * scale);
  canvasEl.height = Math.round(vh * scale);
  const ctx = canvasEl.getContext("2d");
  ctx.drawImage(videoEl, 0, 0, canvasEl.width, canvasEl.height);

  return new Promise((resolve, reject) => {
    canvasEl.toBlob(
      (blob) => blob ? resolve(blob) : reject(new Error("Canvas capture failed")),
      "image/jpeg", 0.8,
    );
  });
}

// ---------------------------------------------------------------------------
// Progress animation (3 steps, 1.8 s each)
// ---------------------------------------------------------------------------
const STEP_DURATION_MS = 1800;
const ORB_FADE_MS = 450; // keep in sync with .frozen-orb-slot transition-duration

let activeOrbSlot = 0;

function setOrbState(state, label) {
  const nextEl = frozenOrbSlots[1 - activeOrbSlot];
  const prevEl = frozenOrbSlots[activeOrbSlot];

  renderOrb(nextEl, state, { size: 64, theme: "dark" });
  nextEl.classList.add("visible");
  prevEl.classList.remove("visible");
  setTimeout(() => clearOrb(prevEl), ORB_FADE_MS);
  activeOrbSlot = 1 - activeOrbSlot;

  frozenOrbLabel.style.opacity = "0";
  setTimeout(() => {
    frozenOrbLabel.textContent = label;
    frozenOrbLabel.style.opacity = "1";
  }, ORB_FADE_MS / 2);
}

function startProgressAnimation() {
  let current = 0;

  function advance() {
    if (current < STEP_LABELS.length) {
      setOrbState(STEP_ORB_STATES[current], STEP_LABELS[current]);
      triggerHaptic("light");
      current++;
      if (current < STEP_LABELS.length) setTimeout(advance, STEP_DURATION_MS);
    }
  }
  advance();
  return new Promise((resolve) => setTimeout(resolve, STEP_DURATION_MS * STEP_LABELS.length));
}

// ---------------------------------------------------------------------------
// Before/after slider
// ---------------------------------------------------------------------------
let isDragging = false;

function initSlider() {
  compareWrapper.addEventListener("pointerdown", onSliderDown, { passive: true });
  document.addEventListener("pointermove", onSliderMove, { passive: true });
  document.addEventListener("pointerup",   () => isDragging = false);
}

function onSliderDown(e) {
  isDragging = true;
  updateSplit(e.clientX);
}

function onSliderMove(e) {
  if (!isDragging) return;
  updateSplit(e.clientX);
}

function updateSplit(clientX) {
  const rect = compareWrapper.getBoundingClientRect();
  const pct = Math.max(5, Math.min(95, ((clientX - rect.left) / rect.width) * 100));
  compareWrapper.style.setProperty("--split", `${pct}%`);
  afterSide.style.clipPath = `inset(0 0 0 ${pct}%)`;
  compareDivider.style.left = `${pct}%`;
}

function resetSplit() {
  const pct = 50;
  compareWrapper.style.setProperty("--split", `${pct}%`);
  afterSide.style.clipPath = `inset(0 0 0 ${pct}%)`;
  compareDivider.style.left = `${pct}%`;
}

// ---------------------------------------------------------------------------
// Sound cue
// ---------------------------------------------------------------------------
function playResultChime() {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.type = "sine";
    osc.frequency.setValueAtTime(880, ctx.currentTime);
    osc.frequency.exponentialRampToValueAtTime(1320, ctx.currentTime + 0.12);
    gain.gain.setValueAtTime(0.18, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.35);
    osc.start(ctx.currentTime);
    osc.stop(ctx.currentTime + 0.35);
  } catch { /* AudioContext not available */ }
}

// ---------------------------------------------------------------------------
// Toast
// ---------------------------------------------------------------------------
let toastTimer = null;
function showToast(msg, isError = false, duration = 3000) {
  clearTimeout(toastTimer);
  toastEl.textContent = msg;
  toastEl.className = `toast visible${isError ? " error" : ""}`;
  toastTimer = setTimeout(() => {
    toastEl.className = "toast";
  }, duration);
}

// ---------------------------------------------------------------------------
// Render result
// ---------------------------------------------------------------------------
function renderResult(data) {
  imgBefore.src = capturedDataUrl;
  imgAfter.src  = `data:image/jpeg;base64,${data.edited_image}`;

  const sb = data.score_before;
  const sa = data.score_after;
  scoreBefore.textContent = sb.toFixed(2);
  scoreAfter.textContent  = sa.toFixed(2);

  const diff = sa - sb;
  const pct  = sb > 0 ? (diff / sb) * 100 : 0;
  const sign  = diff >= 0 ? "+" : "";

  if (diff > 0.005) {
    improvementChip.className = "improvement-chip";
    improvementChip.textContent = `${sign}${pct.toFixed(1)}%  (${sign}${diff.toFixed(2)})`;
  } else if (diff < -0.005) {
    improvementChip.className = "improvement-chip negative";
    improvementChip.textContent = `${pct.toFixed(1)}%  (${diff.toFixed(2)})`;
  } else {
    improvementChip.className = "improvement-chip neutral";
    improvementChip.textContent = "Score unchanged";
  }

  latencyLabel.textContent = `${(data.latency_ms / 1000).toFixed(1)}s`;
  feedbackText.textContent = data.feedback;
  resetSplit();
}

// ---------------------------------------------------------------------------
// Leaderboard
// ---------------------------------------------------------------------------
function rankClass(rank) {
  if (rank === 1) return "gold";
  if (rank === 2) return "silver";
  if (rank === 3) return "bronze";
  return "";
}

async function fetchAndRenderLeaderboard(listEl, subtitleEl) {
  try {
    const res  = await fetch(LEADERBOARD_ENDPOINT, { signal: AbortSignal.timeout(4000) });
    const data = await res.json();

    if (subtitleEl) {
      subtitleEl.textContent = `${data.total_edits} total edit${data.total_edits !== 1 ? "s" : ""} — top ${data.entries.length} by improvement`;
    }

    editCountBadge.textContent = data.total_edits;
    editCountBadge.hidden = data.total_edits === 0;

    listEl.innerHTML = "";
    if (data.entries.length === 0) {
      const li = document.createElement("li");
      li.textContent = "No edits yet — be the first!";
      li.style.cssText = "color:var(--muted);font-size:14px;padding:8px 0;";
      listEl.appendChild(li);
      return;
    }

    for (const e of data.entries) {
      const li = document.createElement("li");
      li.className = "leaderboard-item";
      const impClass = e.improvement_pct >= 0 ? "" : "negative";
      const impSign  = e.improvement_pct >= 0 ? "+" : "";
      li.innerHTML = `
        <span class="lb-rank ${rankClass(e.rank)}">#${e.rank}</span>
        <span class="lb-scores">
          ${e.score_before.toFixed(2)}
          <span class="lb-arrow">→</span>
          ${e.score_after.toFixed(2)}
        </span>
        <span class="lb-improvement ${impClass}">${impSign}${e.improvement_pct.toFixed(1)}%</span>
      `;
      listEl.appendChild(li);
    }
  } catch {
    if (subtitleEl) subtitleEl.textContent = "Could not load leaderboard.";
  }
}

function openLeaderboard() {
  leaderboardBackdrop.hidden = false;
  fetchAndRenderLeaderboard(leaderboardList, leaderboardSubtitle);
  lbPollTimer = setInterval(() => {
    fetchAndRenderLeaderboard(leaderboardList, leaderboardSubtitle);
  }, 15_000);
}

function closeLeaderboardModal() {
  leaderboardBackdrop.hidden = true;
  clearInterval(lbPollTimer);
}

// ---------------------------------------------------------------------------
// Main capture flow
// ---------------------------------------------------------------------------
async function handleCapture() {
  triggerHaptic("selection");
  captureBtn.disabled = true;

  let blob;
  try {
    blob = await captureFrame();
  } catch (err) {
    showToast("Could not capture frame", true);
    captureBtn.disabled = !serverOnline;
    return;
  }

  capturedBlob = blob;
  capturedDataUrl = URL.createObjectURL(blob);
  frozenPreview.src = capturedDataUrl;

  setState("processing");
  const minWait = startProgressAnimation();

  const formData = new FormData();
  formData.append("image", blob, "capture.jpg");

  let data;
  try {
    const [res] = await Promise.all([
      fetch(EDIT_ENDPOINT, { method: "POST", body: formData }),
      minWait,
    ]);
    if (!res.ok) {
      const detail = await res.json().then((j) => j.detail).catch(() => res.statusText);
      throw new Error(detail || `HTTP ${res.status}`);
    }
    data = await res.json();
  } catch (err) {
    showToast(err.message || "Request failed", true, 4000);
    triggerHaptic("error");
    setState("idle");
    captureBtn.disabled = !serverOnline;
    return;
  }

  renderResult(data);
  setState("result");
  playResultChime();
  triggerHaptic("success");

  // Refresh leaderboard badge count
  fetchAndRenderLeaderboard({ innerHTML: "" }, null).catch(() => {});
  fetch(LEADERBOARD_ENDPOINT, { signal: AbortSignal.timeout(4000) })
    .then((r) => r.json())
    .then((d) => {
      editCountBadge.textContent = d.total_edits;
      editCountBadge.hidden = d.total_edits === 0;
    })
    .catch(() => {});
}

// ---------------------------------------------------------------------------
// State machine
// ---------------------------------------------------------------------------
function setState(state) {
  bodyEl.className = bodyEl.className
    .split(" ")
    .filter((c) => !c.startsWith("state-"))
    .join(" ");
  bodyEl.classList.add(`state-${state}`);
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
function init() {
  if (isDisplayMode) return;

  initCamera();
  initSlider();
  checkHealth();
  setInterval(checkHealth, 15_000);

  captureBtn.addEventListener("click", handleCapture);
  retryBtn.addEventListener("click", () => {
    setState("idle");
    captureBtn.disabled = !serverOnline;
    if (capturedDataUrl) {
      URL.revokeObjectURL(capturedDataUrl);
      capturedDataUrl = null;
    }
  });
  leaderboardBtn.addEventListener("click", () => {
    triggerHaptic("light");
    openLeaderboard();
  });
  closeLeaderboard.addEventListener("click", closeLeaderboardModal);
  leaderboardBackdrop.addEventListener("click", (e) => {
    if (e.target === leaderboardBackdrop) closeLeaderboardModal();
  });
}

init();
