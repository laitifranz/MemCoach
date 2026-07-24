// Thin wrapper mounting the `thinking-orbs` React component into a plain DOM
// node, so the rest of the app can stay framework-free.
// https://github.com/Jakubantalik/thinking-orbs
import React from "https://esm.sh/react@18.3.1";
import { createRoot } from "https://esm.sh/react-dom@18.3.1/client";
import { ThinkingOrb } from "https://esm.sh/thinking-orbs@0.1.1?deps=react@18.3.1,react-dom@18.3.1";

const roots = new Map();

export function renderOrb(container, state, { size = 20, theme = "dark", paused = false } = {}) {
  if (!container) return;
  let root = roots.get(container);
  if (!root) {
    root = createRoot(container);
    roots.set(container, root);
  }
  root.render(React.createElement(ThinkingOrb, { state, size, theme, paused }));
}

export function clearOrb(container) {
  const root = roots.get(container);
  if (!root) return;
  root.unmount();
  roots.delete(container);
}
