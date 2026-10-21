"""
web/dashboard.py — the same engine, in a browser.

Why: examiners / teammates may not have a desktop GUI (or are on a lab PC),
and a browser dashboard makes the demo presentable on one screen.

Run:
    python web/dashboard.py
then open http://localhost:5000

Endpoints
    GET  /                dashboard (no CDN, works offline)
    GET  /stream          MJPEG stream of the 2x2 grid
    GET  /api/state       JSON snapshot (counts, lights, reason)
    POST /api/trigger/NORTH   simulate an ambulance on that lane
    POST /api/clear           clear simulated emergencies
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import cv2
from flask import Flask, Response, jsonify, request

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg  # noqa: E402
from smart_intersection import SmartIntersectionAI  # noqa: E402

app = Flask(__name__)


class EngineRunner(threading.Thread):
    """Runs the engine in the background and keeps the latest JPEG."""

    daemon = True

    def __init__(self, engine: SmartIntersectionAI, target_fps: float = 10.0) -> None:
        super().__init__(name="engine")
        self.engine = engine
        self.target_dt = 1.0 / target_fps
        self.lock = threading.Lock()
        self.jpeg: bytes | None = None
        self.seq = 0
        self.stop_flag = threading.Event()

    def run(self) -> None:
        while not self.stop_flag.is_set():
            t0 = time.time()
            try:
                with self.lock:
                    grid = self.engine.step()
                ok, buf = cv2.imencode(".jpg", grid, [cv2.IMWRITE_JPEG_QUALITY, 78])
                if ok:
                    self.jpeg = buf.tobytes()
                    self.seq += 1
            except Exception as exc:                      # keep the server alive
                print(f"[runner] error: {exc}")
                time.sleep(0.5)
            time.sleep(max(0.0, self.target_dt - (time.time() - t0)))

    def state(self) -> dict:
        eng = self.engine
        d = eng.decision
        return {
            "green": d.green,
            "reason": d.reason,
            "emergency": bool(d.emergency),
            "hold": round(getattr(d, "hold_remaining", 0.0), 1),
            "fps": round(eng.fps, 1),
            "mode": eng.mode,
            "weights": eng.weights.name,
            "frame": eng.frame_index,
            "lanes": [
                {
                    "name": l.name,
                    "count": l.count,
                    "priority": l.priority_count,
                    "light": l.light,
                    "simulated": l.simulated,
                }
                for l in eng.lanes
            ],
        }


runner: EngineRunner | None = None


def get_runner() -> EngineRunner:
    global runner
    if runner is None or not runner.is_alive():
        engine = SmartIntersectionAI(verbose=True)
        runner = EngineRunner(engine)
        runner.start()
    return runner


# --------------------------------------------------------------------------- #
PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Smart Intersection AI</title>
<style>
  :root{--bg:#0d0f14;--card:#171a21;--line:#262b36;--txt:#eef1f6;--dim:#8d95a5;
        --green:#3cdc3c;--red:#e62828;--pink:#ff3cdc;}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--txt);
       font-family:ui-sans-serif,system-ui,"Segoe UI",Roboto,Arial,sans-serif}
  header{padding:18px 24px;border-bottom:1px solid var(--line);display:flex;
         align-items:center;gap:14px;flex-wrap:wrap}
  header h1{font-size:19px;margin:0;letter-spacing:.4px}
  .pill{font-size:12px;color:var(--dim);border:1px solid var(--line);
        padding:3px 9px;border-radius:999px}
  main{max-width:1180px;margin:0 auto;padding:20px 24px 60px}
  .wrap{display:grid;grid-template-columns:minmax(0,1.55fr) minmax(280px,1fr);gap:18px}
  @media(max-width:900px){.wrap{grid-template-columns:1fr}}
  .card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px}
  .card h2{font-size:13px;text-transform:uppercase;letter-spacing:.12em;
           color:var(--dim);margin:0 0 12px}
  img.stream{width:100%;border-radius:10px;display:block;background:#000}
  .big{font-size:30px;font-weight:700;margin:0 0 4px}
  .reason{color:var(--dim);font-size:13px;min-height:18px}
  .stat{display:flex;justify-content:space-between;padding:7px 0;
        border-bottom:1px dashed var(--line);font-size:13px}
  .stat:last-child{border-bottom:0}
  .stat b{font-size:13px}
  .g{color:var(--green)} .r{color:var(--red)} .p{color:var(--pink)}
  .btns{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:10px}
  button{background:#222734;color:var(--txt);border:1px solid var(--line);
         border-radius:9px;padding:10px 8px;font-size:13px;cursor:pointer}
  button:hover{background:#2c3342;border-color:#3b4459}
  button.warn{border-color:#5b2b45}
  footer{color:var(--dim);font-size:12px;margin-top:18px;line-height:1.7}
  code{background:#101319;padding:1px 5px;border-radius:5px;font-size:12px}
</style></head>
<body>
<header>
  <h1>Smart Intersection AI</h1>
  <span class="pill" id="mode">mode</span>
  <span class="pill" id="weights">weights</span>
  <span class="pill" id="fps">0 fps</span>
</header>
<main>
  <div class="wrap">
    <div class="card">
      <h2>Live 4-lane dashboard</h2>
      <img class="stream" src="/stream" alt="live grid"/>
      <div class="btns">
        <button onclick="post('/api/trigger/NORTH')">Simulate ambulance &middot; NORTH</button>
        <button onclick="post('/api/trigger/SOUTH')">Simulate ambulance &middot; SOUTH</button>
        <button onclick="post('/api/trigger/EAST')">Simulate ambulance &middot; EAST</button>
        <button onclick="post('/api/trigger/WEST')">Simulate ambulance &middot; WEST</button>
      </div>
      <div class="btns" style="grid-template-columns:1fr">
        <button class="warn" onclick="post('/api/clear')">Clear simulated emergencies</button>
      </div>
    </div>
    <div>
      <div class="card">
        <h2>Decision</h2>
        <p class="big" id="green">&mdash;</p>
        <div class="reason" id="reason"></div>
        <div class="stat"><span>Emergency override</span><b id="emg">no</b></div>
        <div class="stat"><span>Hold remaining</span><b id="hold">0.0 s</b></div>
        <div class="stat"><span>Frame</span><b id="frame">0</b></div>
      </div>
      <div class="card" style="margin-top:18px">
        <h2>Lanes</h2>
        <div id="lanes"></div>
      </div>
      <footer>
        Simulation is used because the shipped clips contain no ambulance and
        pretrained COCO has no ambulance class. Drop a trained
        <code>models/best.pt</code> in and real detections take over.
      </footer>
    </div>
  </div>
</main>
<script>
async function post(url){await fetch(url,{method:'POST'});}
async function poll(){
  try{
    const s = await (await fetch('/api/state')).json();
    document.getElementById('green').innerHTML =
      `<span class="${s.emergency?'p':'g'}">${s.green} GREEN</span>`;
    document.getElementById('reason').textContent = s.reason;
    document.getElementById('emg').innerHTML =
      s.emergency ? '<span class="p">YES</span>' : 'no';
    document.getElementById('hold').textContent = s.hold + ' s';
    document.getElementById('frame').textContent = s.frame;
    document.getElementById('fps').textContent = s.fps + ' fps';
    document.getElementById('mode').textContent = s.mode;
    document.getElementById('weights').textContent = s.weights;
    document.getElementById('lanes').innerHTML = s.lanes.map(l=>`
      <div class="stat">
        <span>${l.name}${l.simulated?' <span class="p">(sim)</span>':''}</span>
        <b class="${l.light==='GREEN'?'g':'r'}">${l.light}</b>
        <span>${l.count} veh${l.priority?` · <span class="p">${l.priority} priority</span>`:''}</span>
      </div>`).join('');
  }catch(e){}
}
setInterval(poll, 1000); poll();
</script>
</body></html>"""


@app.get("/")
def index() -> str:
    get_runner()
    return PAGE


@app.get("/stream")
def stream():
    r = get_runner()

    def gen():
        last = -1
        while True:
            if r.jpeg is not None and r.seq != last:
                last = r.seq
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + r.jpeg + b"\r\n")
            time.sleep(0.05)

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/state")
def state():
    return jsonify(get_runner().state())


@app.post("/api/trigger/<lane>")
def trigger(lane: str):
    lane = lane.upper()
    if lane not in cfg.LANE_NAMES:
        return jsonify({"ok": False, "error": f"unknown lane {lane}"}), 400
    r = get_runner()
    r.engine.simulator.trigger(lane, seconds=6.0, now=r.engine.clock())
    return jsonify({"ok": True, "lane": lane})


@app.post("/api/clear")
def clear():
    get_runner().engine.simulator.clear()
    return jsonify({"ok": True})


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=int(5000))
    ap.add_argument("--weights", default=None)
    args = ap.parse_args()

    if args.weights:
        cfg.CUSTOM_WEIGHTS = Path(args.weights)
        cfg.DEFAULT_WEIGHTS = Path(args.weights)

    print(f"[web] starting on http://localhost:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
