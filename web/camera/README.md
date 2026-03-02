# MemCoach Camera MVP

Minimal mobile-friendly web client that captures a camera frame, compresses it, and sends it to the FastAPI endpoints declared in `src/api/app.py`.

## Quick Start

1. Start the FastAPI server (default host/port defined in `config/api/server.yaml`). If you leave the default `web/camera` directory in place, the app is automatically mounted at `http://<host>:<port>/camera`, so you can skip running a separate static server.
2. (Optional) Serve this folder standalone if you prefer:
   ```bash
   cd web/camera
   python -m http.server 8080
   ```
3. On your phone, connect to the same Wi-Fi and open either `http://<backend-host>:<backend-port>/camera` (same-origin setup) or your standalone server URL. Grant camera permissions.
4. If you use the standalone server served over plain HTTP, the UI automatically redirects API calls to port 8000 on the same host. If served over HTTPS without a `?api` override, the UI falls back to the same origin. You can always override the API target by appending `?api=http://<server-host>:<port>` to the URL (required when the UI is loaded via ngrok or another tunnel pointing to a different host).

## Usage Tips

- Health chip turns green when `/health` responds with `{ "status": "ok" }` and re-checks every 30 seconds automatically.
- `Mem Score` calls `POST /score`; `Score + Feedback` calls `POST /score-feedback`.
- The feedback card shows a truncated inline preview (up to 200 characters); tap it to open a dialog with the full text.
- A spinner covers the preview while uploads are running.
- The **Latest Results** section tracks your last 10 runs, showing score, latency, mode label, timestamp, and delta vs. the previous score (both absolute and percentage).
- Before uploading, the UI downscales the frame to a maximum of 1024 px on the longest edge and JPEG-compresses it at 0.8 quality. The server accepts JPEG, PNG, and WebP.
- The camera defaults to the rear-facing lens (`facingMode: environment`).

## Deployment Options

- Default setup already mounts `web/camera` under `/camera`, so the UI and API share the same origin (required for ngrok or other tunnels).
- If you still need a standalone host, `python -m http.server` or `npm serve` works; just be mindful of mixed-content restrictions (HTTPS page calling HTTP API).
- Users can add the page to the home screen via the mobile browser menu for an app-like feel; a full PWA manifest + service worker can be layered later if needed.
