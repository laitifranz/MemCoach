// adapted from https://github.com/lochie/web-haptics/tree/main/packages/web-haptics/src/lib/web-haptics
// EXPERIMENTAL IMPLEMENTATION OF HAPTICS 

const MAX_HAPTIC_PHASE_MS = 1000;
const PWM_CYCLE_MS = 20;
const TOGGLE_MIN = 16;
const TOGGLE_MAX = 184;

const HAPTIC_PATTERNS = {
  success: [
    { duration: 30, intensity: 0.5 },
    { delay: 60, duration: 40, intensity: 1 },
  ],
  warning: [
    { duration: 40, intensity: 0.8 },
    { delay: 100, duration: 40, intensity: 0.6 },
  ],
  error: [
    { duration: 40, intensity: 0.9 },
    { delay: 40, duration: 40, intensity: 0.9 },
    { delay: 40, duration: 40, intensity: 0.9 },
  ],
  light: [{ duration: 15, intensity: 0.4 }],
  selection: [{ duration: 8, intensity: 0.3 }],
};

const HAPTIC_VIBRATE_SUPPORTED =
  typeof navigator !== "undefined" && typeof navigator.vibrate === "function";

// Hidden switch checkbox — the iOS Safari haptic trick.
// Clicking a <label> for a switch-type checkbox triggers a native haptic
// on iOS Safari from within a user gesture, without any special permission.
let hapticLabel = null;
let hapticRafId = null;

function ensureHapticDOM() {
  if (hapticLabel) return;
  const checkbox = document.createElement("input");
  checkbox.type = "checkbox";
  checkbox.setAttribute("switch", "");
  checkbox.id = "haptic-switch";
  checkbox.style.cssText = "all:initial;appearance:auto;display:none";

  const label = document.createElement("label");
  label.setAttribute("for", "haptic-switch");
  label.style.cssText = "display:none;position:fixed;pointer-events:none";
  label.appendChild(checkbox);
  document.body.appendChild(label);
  hapticLabel = label;
}

function normalizeHapticInput(input) {
  if (typeof input === "number") {
    return [{ duration: input }];
  }
  if (typeof input === "string") {
    const preset = HAPTIC_PATTERNS[input];
    return preset ? preset.map((v) => ({ ...v })) : null;
  }
  if (Array.isArray(input)) {
    if (input.length === 0) return [];
    if (typeof input[0] === "number") {
      const vibrations = [];
      for (let i = 0; i < input.length; i += 2) {
        const duration = input[i];
        if (!Number.isFinite(duration)) return null;
        const delay = i > 0 ? input[i - 1] : 0;
        const step = { duration };
        if (delay > 0) step.delay = delay;
        vibrations.push(step);
      }
      return vibrations;
    }
    return input.map((v) => ({ ...v }));
  }
  if (input && Array.isArray(input.pattern)) {
    return input.pattern.map((v) => ({ ...v }));
  }
  return null;
}

function modulateHaptic(duration, intensity) {
  if (intensity >= 1) return [duration];
  if (intensity <= 0) return [];
  const onTime = Math.max(1, Math.round(PWM_CYCLE_MS * intensity));
  const offTime = PWM_CYCLE_MS - onTime;
  const segments = [];
  let remaining = duration;
  while (remaining >= PWM_CYCLE_MS) {
    segments.push(onTime, offTime);
    remaining -= PWM_CYCLE_MS;
  }
  if (remaining > 0) {
    const remOn = Math.max(1, Math.round(remaining * intensity));
    segments.push(remOn);
    const remOff = remaining - remOn;
    if (remOff > 0) segments.push(remOff);
  }
  return segments;
}

function toVibratePattern(vibrations, defaultIntensity) {
  const pattern = [];
  for (const vib of vibrations) {
    const delay = vib.delay ?? 0;
    const intensity = Math.max(0, Math.min(1, vib.intensity ?? defaultIntensity));
    if (delay > 0) {
      if (pattern.length > 0 && pattern.length % 2 === 0) {
        pattern[pattern.length - 1] += delay;
      } else {
        if (pattern.length === 0) pattern.push(0);
        pattern.push(delay);
      }
    }
    const modulated = modulateHaptic(vib.duration, intensity);
    if (modulated.length === 0) {
      if (vib.duration > 0) {
        if (pattern.length > 0 && pattern.length % 2 === 0) {
          pattern[pattern.length - 1] += vib.duration;
        } else {
          pattern.push(0, vib.duration);
        }
      }
      continue;
    }
    for (const seg of modulated) pattern.push(seg);
  }
  return pattern;
}

// RAF loop that clicks the hidden switch at intervals derived from intensity,
// reproducing the multi-vibration pattern on iOS Safari via native haptics.
function runHapticPattern(vibrations, defaultIntensity, firstClickFired) {
  if (hapticRafId !== null) {
    cancelAnimationFrame(hapticRafId);
    hapticRafId = null;
  }

  const phases = [];
  let cumulative = 0;
  for (const vib of vibrations) {
    const intensity = Math.max(0, Math.min(1, vib.intensity ?? defaultIntensity));
    const delay = vib.delay ?? 0;
    if (delay > 0) {
      cumulative += delay;
      phases.push({ end: cumulative, isOn: false, intensity: 0 });
    }
    cumulative += vib.duration;
    phases.push({ end: cumulative, isOn: true, intensity });
  }
  const totalDuration = cumulative;

  let startTime = 0;
  let lastToggleTime = -1;

  function loop(time) {
    if (startTime === 0) startTime = time;
    const elapsed = time - startTime;
    if (elapsed >= totalDuration) {
      hapticRafId = null;
      return;
    }

    let phase = phases[0];
    for (const p of phases) {
      if (elapsed < p.end) { phase = p; break; }
    }

    if (phase.isOn) {
      const toggleInterval = TOGGLE_MIN + (1 - phase.intensity) * TOGGLE_MAX;
      if (lastToggleTime === -1) {
        lastToggleTime = time;
        if (!firstClickFired) {
          hapticLabel.click();
          firstClickFired = true;
        }
      } else if (time - lastToggleTime >= toggleInterval) {
        hapticLabel.click();
        lastToggleTime = time;
      }
    }

    hapticRafId = requestAnimationFrame(loop);
  }
  hapticRafId = requestAnimationFrame(loop);
}

export function triggerHaptic(input = "light", options = {}) {
  const vibrations = normalizeHapticInput(input);
  if (!vibrations || vibrations.length === 0) return;

  const defaultIntensity = Math.max(0, Math.min(1, options.intensity ?? 0.5));

  const validVibrations = [];
  for (const vib of vibrations) {
    const duration = Math.min(MAX_HAPTIC_PHASE_MS, Number(vib.duration));
    const delayValue = vib.delay === undefined ? undefined : Number(vib.delay);
    if (
      !Number.isFinite(duration) || duration < 0 ||
      (delayValue !== undefined && (!Number.isFinite(delayValue) || delayValue < 0))
    ) return;
    const sanitized = { duration };
    if (delayValue !== undefined) sanitized.delay = delayValue;
    if (vib.intensity !== undefined) sanitized.intensity = Number(vib.intensity);
    validVibrations.push(sanitized);
  }

  // Android / desktop: use Vibration API directly.
  if (HAPTIC_VIBRATE_SUPPORTED) {
    const pattern = toVibratePattern(validVibrations, defaultIntensity);
    if (pattern.length > 0) navigator.vibrate(pattern);
  }

  // iOS Safari (and any browser without navigator.vibrate): click a hidden
  // switch checkbox synchronously from the user gesture, then schedule
  // subsequent clicks via RAF to reproduce multi-vibration patterns.
  if (!HAPTIC_VIBRATE_SUPPORTED) {
    ensureHapticDOM();
    const firstDelay = validVibrations[0]?.delay ?? 0;
    const firstClickFired = firstDelay === 0;
    if (firstClickFired) hapticLabel.click();
    runHapticPattern(validVibrations, defaultIntensity, firstClickFired);
  }
}
