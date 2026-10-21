# 🎤 Viva / Review Q&A — Smart Intersection AI

Answers are written the way you should *say* them: short, honest, no hand-waving.
If you don't know something, say "we measured X, we did not test Y" — examiners respect that far
more than a bluff.

---

## A. The basics

**1. What problem are you solving?**
Fixed-time signals are blind. An ambulance can wait at red while an empty road holds green.
We give the junction a camera and a detector, so the green light follows what is actually on the
road — emergency vehicles first, then the busiest approach.

**2. Why not just use a timer, or loop detectors?**
A timer cannot react to an ambulance. Inductive loops detect *presence*, not *class* — they cannot
tell an ambulance from a car. A camera can do both counting and classification with one sensor that
is often already installed.

**3. What exactly is the input and the output?**
Input: four video feeds (files or camera indices). Output: a signal state per approach
(GREEN/RED), a 2×2 dashboard, a browser view and a per-frame CSV log. In this version the signal
exists on screen; driving real lamps is future work.

**4. Walk me through one frame.**
Decode the four frames → downscale → YOLOv8 inference → per approach: count vehicles and collect
priority detections → controller decides the green approach → renderer draws boxes, lamps and HUD →
log a row → next frame.

---

## B. The AI part

**5. Which model and why?**
YOLOv8n from Ultralytics. Single-stage detector, ~3 M parameters, fast enough for CPU at reduced
frame rate and for a Jetson at video rate. Small model = trains on a free Colab T4 in about an hour.

**6. What happens when the model is wrong — a false positive car?**
Two protections: a confidence threshold (0.35) and, more importantly, the switching margin. A lane
must lead by two vehicles before it takes the green, so a single false detection cannot flip the
light. Emergency decisions are intentionally *not* damped — safety beats smoothness there, and the
worst case of a false emergency is one junction clearing early.

**7. The pretrained model has no "ambulance" class. Did you fake the results?**
No, and the README says so plainly. Pretrained COCO knows car, bus, truck, motorcycle — no
ambulance, no police. So the repo runs in two modes: with COCO weights the counting and
density-based switching are real, and emergency events come from an explicit simulator that is
labelled "SIMULATED AMBULANCE" on screen. With a trained `models/best.pt` the same code path uses
real detections. The decision logic, which is the contribution, is identical in both modes and is
covered by 13 unit tests.

**8. Why do classes work by name and not by index?**
`detector.py` matches class *names* against keywords (`ambulance`, `police`, `fire`, …). If you
retrain with a different class order, or use a public dataset with different indices, it still
works — the index is an implementation detail, the name is the meaning.

**9. What is mAP and which number matters for you?**
Mean average precision over IoU thresholds; AP50 is at IoU 0.5. Overall mAP is misleading here
because cars dominate the data — a model can score 0.85 overall and 0.2 on ambulance. The number
that matters is **per-class AP50 for ambulance**, and `notebooks/train_colab.ipynb` prints exactly
that.

**10. How would you make detection better at night?**
Three things: more night images in the dataset (the honest fix), strong value/saturation
augmentation during training (already configured: `hsv_v=0.5, hsv_s=0.7`), and raising input
resolution for distant vehicles. Also fusing audio siren detection, which is in future work.

**11. Why do you only run detection every second frame?**
Signal decisions change over seconds, not milliseconds. Halving inference frequency halves CPU
load with no practical loss. It is a CLI flag (`--detect-every`), so it can be raised for a GPU
deployment.

---

## C. The decision logic (they will test you here)

**12. How do you stop the light flickering between two lanes?**
Two mechanisms: a minimum green time (3 s) and a hysteresis margin — a lane must beat the current
lane by 2 vehicles before taking over. Both are unit-tested
(`test_min_green_prevents_flicker`, `test_small_lead_does_not_steal_the_green`).

**13. Why hold the green after the ambulance leaves?**
An ambulance is followed by other vehicles — a convoy, or just queued traffic that needs to clear.
If the light flips the instant the box disappears, the junction deadlocks. Default hold is 6 s.

**14. Two lanes have an ambulance at the same time. What happens?**
The lane with more priority vehicles wins; ties break deterministically in lane order (N, S, E, W)
so behaviour is reproducible. Tested in `test_two_emergencies_more_vehicles_wins`.

**15. An ambulance appears during another lane's minimum green. Wait for the timer?**
No. Emergency preemption bypasses the minimum green entirely. Tested in
`test_emergency_beats_min_green`.

**16. Empty junction at 3 a.m. — does the light go dark?**
No, it holds the last green. Flickering between empty lanes would be worse for approaching drivers.

**17. Is the decision logic testable without a GPU?**
Yes — that is a design decision, not luck. `controller.py` imports nothing but the standard
library and receives time as an argument, so every rule is a pure function of counts, priority and
`now`. That is why 13 tests run in 0.08 s with no model, no video, no display.

---

## D. Engineering and honesty checks

**18. Did it run on a fresh clone before you started?**
No. `controller.py` pointed at `runs/detect/smart_traffic_model/weights/best.pt`, which was never in
the repo, so it crashed immediately; `main.py` was a single-image script on a photo of toy cars. The
4-lane product described in the README did not exist. Making it run is most of today's work, and
`docs/CHECKLIST.md` documents exactly what was found and what was fixed.

**19. What are your measured numbers?**
On a 2-vCPU CPU-only machine: ~72 ms per YOLOv8n pass at 640 px, ~4.5–5 dashboard frames/s for all
four approaches, decision computation under 1 ms, 5–13 vehicles detected per night frame, 13/13
tests passing. We did not measure on a GPU; we do not claim the 45 FPS that the old README
advertised.

**20. What is the single biggest limitation?**
No trained ambulance detector yet. It needs annotated Indian ambulance imagery and about an hour of
GPU time; the pipeline for it is written and committed (`prepare_dataset.py` → `train.py` / Colab),
so the remaining cost is data and compute, not code.

**21. What would break first in a real deployment?**
Occlusion and camera placement, not the model. A camera that sees the last 30 m of an approach
gives you about 2–3 s of warning at 40 km/h. That is enough for a signal change, but it means camera
geometry has to be designed together with the signal timing.

**22. Is this scalable to a city?**
The per-junction engine is independent, so junctions scale horizontally. What does *not* scale
naively is a green corridor — that needs junction-to-junction coordination (future work item 4).

---

## E. Questions you should ask back / things to volunteer

- "The simulator flag is visible on screen and in the log — we never present simulated events as
  real detections."
- "Every number in the README is measured on the machine we ran it on; the old 98% mAP / 45 FPS
  claims were removed because nobody had measured them."
- "The demo clips are night traffic with no emergency vehicles; that is why the override is
  demonstrated with the simulator."

---

## F. Two-minute demo script

1. `python main.py` — window opens, four lanes detected live, header shows which lane is green and
   *why* ("Busiest lane (11 vehicles)").
2. Point out red lanes are dimmed, the signal lamps, and the per-box class + confidence.
3. Press `1` — NORTH turns green instantly, magenta banner flashes, header says
   "EMERGENCY vehicle on NORTH", every other lane goes red. Say the words: *"one frame, no timer
   involved."*
4. Wait ~6 s — the hold explanation appears in the header, then density control resumes.
5. `Ctrl+C`, then `python web/dashboard.py` → same engine in the browser, buttons trigger the
   ambulance.
6. Close with `python -m unittest discover -s tests -v` — 13 tests green, "this is the part that is
   proven, not just demonstrated."
