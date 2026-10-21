# ✅ Project Checklist — Smart Intersection AI (Priority Traffic Router)

Last updated: **21 Oct 2026** · Repo: `Devesh-tiw/Priority_Traffic_router` @ `2ef72c9`

Legend: ✅ done and verified · 🟡 done but needs your input · ⏳ remaining · ❌ out of scope today

---

## 0. What the repo actually contained when we started

| Item | Reality found in the repo |
|---|---|
| `main.py` | A **single-image** script (`done.jpg` = a photo of toy cars). Not the 4-lane demo the README describes. |
| `controller.py` | 4-lane logic existed, but hardcoded `runs/detect/smart_traffic_model/weights/best.pt` → **crashed on a fresh clone**. Videos `north/south/east/west.mp4` did not exist. |
| `train.py` | Functional, but no dataset anywhere → the custom model was never trained. |
| `data.yaml` | Declared 4 classes (ambulance, police, car, motorcycle) — **no dataset behind it**. |
| `requirements.txt` | `pip freeze` dump of a CUDA machine (60 pinned lines incl. `nvidia-*`, `cuda-toolkit`). `pip install -r` fails or drags ~3 GB on a normal laptop. |
| README | Promised a 2×2 dashboard, `smart_intersection.py`, `traffic.mp4`, `models/`, `demo/`, "98% mAP", "45 FPS" — **almost none of these files existed**, and no numbers were measured. |
| Missing | No `LICENSE` (README said MIT), no `.gitignore`, no tests, no screenshots, no trained weights, no demo video, no dataset. |

**Conclusion:** the *idea* and the decision logic existed. The **runnable 4-lane product did not.**
That is what was finished today.

---

## 1. ✅ Done today (verified, in this repo)

| # | Task | Evidence |
|---|---|---|
| 1 | Fixed the crash: `main.py` now auto-selects `models/best.pt` if present, else the pretrained `yolov8n.pt` | runs on a fresh clone |
| 2 | Rebuilt the 4-lane engine as real modules (`config`, `detector`, `controller`, `smart_intersection`, `main`) | imports cleanly, no hidden paths |
| 3 | Working **2×2 dashboard**: bounding boxes, per-lane traffic lights, HUD, dimmed red lanes | `demo/dashboard.png` |
| 4 | Real video handling: 2 shipped clips → 4 lanes with offset so they don't look duplicated; clips loop forever | `Lane.read()` |
| 5 | **Emergency preemption logic** — ambulance lane green, every other lane red, overrides another lane's green immediately | `controller.py` + tests |
| 6 | **Emergency hold** — lane stays green `--emergency-hold` s after the vehicle leaves (convoy clears the junction) | `Decision.hold_remaining` |
| 7 | **Anti-flicker** — min-green timer + `--switch-margin` hysteresis | 13 unit tests |
| 8 | Classify by **class name keywords**, so any trained model with `ambulance`/`police`/`fire` in the class names works | `detector.py` |
| 9 | **Emergency simulation** for demos (`--simulate NORTH:3:7`, hotkeys `1–4`, dashboard buttons) — honest, labelled on screen | `demo/emergency.png` |
| 10 | Demo **video rendered** (20 s, 500 frames) + screenshots + per-frame CSV decision log | `demo/` |
| 11 | **13 unit tests** for decision logic (emergency, hold, density, fairness, hysteresis, simulator clock) — all pass without torch | `python -m unittest discover -s tests` |
| 12 | **Browser dashboard** (Flask + MJPEG, no CDN, offline-safe) with manual ambulance buttons | `web/dashboard.py` |
| 13 | **Training pipeline**: `prepare_dataset.py` (frames → auto pre-label → split → class balance) and rewritten `train.py` (auto device, auto-copies `best.pt`) | runnable |
| 14 | **Colab notebook** for free-GPU training | `notebooks/train_colab.ipynb` |
| 15 | Sanitised `requirements.txt` (works on CPU laptops), added `LICENSE` (MIT) and `.gitignore` | `pip install -r` is now sane |
| 16 | README rewritten: real status table, measured numbers, honest limitations | `README.md` |
| 17 | Performance work: frame caching, `--detect-every`, dim/label filtering → usable on a 2-vCPU CPU | measured in README |

---

## 2. ⏳ Remaining — and the exact order to do it in

### Step A — Push today's work (15 min) 🟡 *needs you*
```bash
git add -A && git commit -m "feat: working 4-lane dashboard, tested decision logic, demo, training pipeline"
git push
```
Watch out: `Traffic_*.mp4` are 35 MB + 8 MB. Keep them (they're already in history) but
`demo/demo.mp4` (~16 MB) is fine too. Don't add `runs/` or `dataset/`.

### Step B — Label + train the ambulance model (2–4 h, mostly waiting) 🟡 *the only real gap*
Order matters here — do **not** start training before labeling enough ambulances:

1. `python prepare_dataset.py frames --video Traffic_east_west.mp4 --out dataset/raw/images`
   (+ the north/south clip; optionally 200–600 stills from YouTube dashcam night traffic)
2. Get ambulance images. Fastest: a Roboflow ambulance/emergency-vehicle dataset
   (search "ambulance detection"), download in **YOLOv8** format. Aim for ≥ 300 ambulance
   instances including Indian 108/EMRI vans, night shots, rear and side views.
3. `python prepare_dataset.py prelabel` → auto-labels car/bus/truck/motorcycle so you only draw
   ambulance + police boxes in [labelImg](https://github.com/HumanSignal/labelImg) or Roboflow.
4. `python prepare_dataset.py split` → prints class balance. **Ambulance boxes must not be 0.**
5. Train on Colab (free T4): `notebooks/train_colab.ipynb`, 100 epochs, 40–90 min.
6. Download `best.pt` → `models/best.pt` → `python main.py`. No flags needed; the simulation
   becomes optional.
7. Record the numbers you actually get: per-class AP50 for `ambulance`, and the new fps.

### Step C — Submission pack (1 h) ✅ *report drafted*
- `docs/REPORT.docx` — full report (abstract → results → future work).
- `docs/VIVA_QA.md` — 20 questions you will be asked, with answers.
- Screenshots: `demo/dashboard.png`, `demo/emergency.png`, `demo/demo.mp4`.
- If the examiner asks for a live demo on a machine without a display → `web/dashboard.py`.

### Step D — Optional extras (only if time is left) ❌
- ESP32 / Raspberry Pi LED over serial (`pyserial`) — the "hardware" slide.
- Siren/audio fusion (`librosa`) to catch an ambulance the camera misses.
- Multi-junction green wave.

---

## 3. Definition of "done today"

- [x] `python main.py` runs on a clean clone with only `requirements.txt` installed
- [x] Emergency override is visible, explained on screen, and unit-tested
- [x] A demo video + screenshots exist (no more "add screenshot here" placeholders)
- [x] Training path exists that needs only *data + time*, not more code
- [x] README tells the truth about what works and what doesn't
- [ ] `models/best.pt` trained — this is a GPU/annotation job, not a code job (Step B)
- [ ] Pushed to GitHub (Step A)

---

## 3b. Handover: how to get this into your repo

This workspace copy is **not** a git repo (the `.git` folder was removed to keep the download
small). The archive `Priority_Traffic_router_delivery.zip` next to this folder contains every new
and changed file — it deliberately leaves out `Traffic_*.mp4` and `yolov8n.pt` because those are
already in your GitHub repo.

```bash
# 1. unzip next to your existing clone, then copy across
unzip Priority_Traffic_router_delivery.zip
cp -r Priority_Traffic_router/*  /path/to/your/clone/     # overlays new + changed files

# 2. sanity check inside your clone
python -m unittest discover -s tests -v      # 13 tests must pass
python main.py --frames 5 --headless         # must not crash

# 3. commit
git add -A
git status                                   # expect: modified main.py, controller.py,
                                             # train.py, requirements.txt, README.md, data.yaml?
                                             # new: config.py detector.py smart_intersection.py
                                             #      prepare_dataset.py tests/ web/ docs/ demo/
                                             #      notebooks/ LICENSE .gitignore
git commit -m "feat: working 4-lane dashboard, tested decision logic, training pipeline, demo"
git push
```

Files you should **not** commit: `runs/`, `dataset/`, `__pycache__/`, `models/*.pt`
(all covered by the new `.gitignore`).

## 4. If you only have 3 hours today

1. **20 min** — Step A (push) + open `demo/demo.mp4` and confirm it plays.
2. **40 min** — Roboflow ambulance dataset download, merge with 100 frames of your clips,
   `prepare_dataset.py split`, check the class balance numbers.
3. **10 min** — start Colab training (Step B.5) and let it run in the background.
4. **40 min** — while it trains: read `docs/VIVA_QA.md`, fill your name/roll no. in `REPORT.docx`.
5. **20 min** — download `best.pt`, drop in `models/`, run `python main.py`, record a 30 s
   screen capture with a real ambulance. That video is your final demo.
