#!/usr/bin/env python3
"""
build_report.py — generates docs/REPORT.docx (college submission report).

Run:  python docs/build_report.py
Edit the CONFIG block (names, course, guide) then re-run to regenerate.
"""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "REPORT.docx"

# --------------------------------------------------------------------------- #
CONFIG = {
    "title": "Smart Intersection AI — Priority Traffic Router",
    "subtitle": "Camera-based emergency vehicle preemption for signalised intersections",
    "course": "B.Tech Artificial Intelligence Engineering",
    "institution": "Amrita Vishwa Vidyapeetham, Amaravati",
    "team": [
        ("Devesh Tiwari", "AV.SC.U4AIE25106"),
        ("D. Mani Teja", "AV.SC.U4AIE25103"),
        ("C. Pranweswar", "AV.SC.U4AIE25102"),
    ],
    "guide": "Course Faculty / Project Guide",
    "date": "October 2026",
}

ACCENT = RGBColor(0x1F, 0x6F, 0xEB)
MONO = "Consolas"


# --------------------------------------------------------------------------- #
def shade(cell, hex_colour: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), hex_colour)
    tc_pr.append(shd)


def h(doc, text, level=1, space_before=14):
    p = doc.add_heading(text, level=level)
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(6)
    for run in p.runs:
        run.font.color.rgb = ACCENT
    return p


def para(doc, text, size=11, bold=False, italic=False, align=None, after=6):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    if align:
        p.alignment = align
    p.paragraph_format.space_after = Pt(after)
    return p


def bullets(doc, items, style="List Bullet"):
    for it in items:
        p = doc.add_paragraph(it, style=style)
        p.paragraph_format.space_after = Pt(3)


def code(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.25)
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(8)
    run = p.add_run(text)
    run.font.name = MONO
    run.font.size = Pt(8.5)
    run._element.rPr.rFonts.set(qn("w:eastAsia"), MONO)
    return p


def table(doc, headers, rows, widths=None, header_fill="1F6FEB"):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = t.rows[0].cells
    for i, text in enumerate(headers):
        hdr[i].text = ""
        p = hdr[i].paragraphs[0]
        run = p.add_run(text)
        run.bold = True
        run.font.size = Pt(9.5)
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        shade(hdr[i], header_fill)
    for row in rows:
        cells = t.add_row().cells
        for i, text in enumerate(row):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            run = p.add_run(str(text))
            run.font.size = Pt(9.5)
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Inches(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def figure(doc, path, caption, width=6.2):
    path = ROOT / path
    if not path.exists():
        para(doc, f"[missing image: {path}]", italic=True)
        return
    doc.add_picture(str(path), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap = para(doc, caption, size=9, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, after=10)
    cap.runs[0].font.color.rgb = RGBColor(0x55, 0x55, 0x55)


# --------------------------------------------------------------------------- #
def build() -> None:
    doc = Document()

    # base style
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)
    style.paragraph_format.line_spacing = 1.15

    for section in doc.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.9)
        section.right_margin = Inches(0.9)

    # ---------------- title page ---------------- #
    for _ in range(2):
        doc.add_paragraph()
    para(doc, CONFIG["institution"].upper(), size=13, bold=True,
         align=WD_ALIGN_PARAGRAPH.CENTER, after=2)
    para(doc, CONFIG["course"], size=11, align=WD_ALIGN_PARAGRAPH.CENTER, after=28)
    t = para(doc, CONFIG["title"], size=24, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, after=6)
    for run in t.runs:
        run.font.color.rgb = ACCENT
    para(doc, CONFIG["subtitle"], size=12, italic=True,
         align=WD_ALIGN_PARAGRAPH.CENTER, after=30)
    para(doc, "PROJECT REPORT", size=11, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, after=24)
    para(doc, "Submitted by", size=10, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, after=8)
    for name, roll in CONFIG["team"]:
        para(doc, f"{name}  —  {roll}", size=12, bold=True,
             align=WD_ALIGN_PARAGRAPH.CENTER, after=2)
    doc.add_paragraph()
    para(doc, f"Under the guidance of {CONFIG['guide']}", size=10, italic=True,
         align=WD_ALIGN_PARAGRAPH.CENTER, after=24)
    para(doc, CONFIG["date"], size=11, align=WD_ALIGN_PARAGRAPH.CENTER)
    doc.paragraphs[-1].add_run().add_break(WD_BREAK.PAGE)

    # ---------------- abstract ---------------- #
    h(doc, "Abstract", level=1, space_before=0)
    para(doc,
         "Fixed-time traffic signals are blind: they cycle red and green on a timer regardless of "
         "what is happening on the road. For an ambulance carrying a critical patient this costs "
         "minutes that cannot be recovered, and for ordinary traffic it wastes fuel at red lights "
         "with no vehicles in front of them. This project gives a four-way intersection eyes and a "
         "brain: four camera feeds are processed by a YOLOv8 object detector, and a decision module "
         "selects the green light every frame. If an emergency vehicle (ambulance, police or fire "
         "service) is detected in any approach, that approach is given green immediately and every "
         "other approach is held at red, overriding whatever green was active. When no emergency "
         "vehicle is present, the green goes to the approach with the highest vehicle density, "
         "subject to a minimum green time and a switching margin that prevent flicker. After the "
         "emergency vehicle leaves the frame the approach holds green long enough for a following "
         "convoy to clear the junction.")
    para(doc,
         "The system is implemented in Python with YOLOv8, OpenCV and NumPy, rendered as a live "
         "2x2 dashboard, exposed through a browser dashboard, and instrumented with a thirteen-case "
         "unit-test suite for the decision logic. A complete training pipeline "
         "(frame extraction, semi-automatic pre-labelling, dataset validation, transfer learning) "
         "is included so that the detector can be specialised from the generic COCO classes to "
         "Indian ambulance and police vehicles. The report is explicit about which parts are "
         "measured, which are simulated, and what remains for future work.")

    h(doc, "Keywords", level=2, space_before=8)
    para(doc, "Intelligent Transportation Systems; YOLOv8; emergency vehicle preemption; "
              "object detection; smart traffic signal; OpenCV; computer vision; transfer learning.",
         italic=True)

    # ---------------- problem ---------------- #
    h(doc, "1. Problem Statement", level=1)
    para(doc, "Signalised intersections in Indian cities are timed, not sensed. Four consequences "
              "follow, and they are the requirements this project answers:")
    table(doc,
          ["#", "Real situation", "Consequence"],
          [["1", "Ambulance stopped at a red light with traffic packed around it",
            "Several minutes lost per junction; directly affects survival"],
           ["2", "Police or fire vehicle unable to clear an empty road", "Delayed response time"],
           ["3", "An empty approach holds green while a congested approach waits",
            "Fuel, time and emissions wasted"],
           ["4", "Sirens are audible but drivers physically cannot move",
            "Audible alerts are insufficient; the signal itself must change"]],
          widths=[0.35, 3.0, 2.8])

    h(doc, "1.1 Objectives", level=2)
    bullets(doc, [
        "Detect and count vehicles independently in four approach cameras in real time.",
        "Detect emergency vehicles (ambulance / police / fire) and identify their approach.",
        "Preempt the signal for that approach in under one frame, and hold it briefly afterwards.",
        "When no emergency exists, serve the highest-density approach without flickering.",
        "Present all of the above on a single operator dashboard, with an auditable log.",
        "Provide the training pipeline needed to specialise the detector for Indian vehicles.",
    ])

    h(doc, "1.2 Scope and delimitation", level=2)
    para(doc, "This is a software (vision + decision) system. It reads video files or camera "
              "streams and produces signal states and a dashboard. Driving physical lamp hardware, "
              "siren fusion and multi-junction coordination are deliberately out of scope and are "
              "listed as future work. The emergency preemption path is demonstrated with a "
              "simulated emergency event because the two sample clips that ship with the project "
              "contain ordinary night traffic only.")

    # ---------------- literature ---------------- #
    h(doc, "2. Background and Related Work", level=1)
    para(doc, "Classical actuated signals use inductive loop detectors buried in the road, which "
              "detect presence but not vehicle class, and cannot distinguish an ambulance from a "
              "car. Vision-based approaches replace loops with cameras and a detector. Earlier "
              "work used background subtraction and blob counting, which is brittle under the "
              "headlights, shadows and reflections of night traffic. Single-stage CNN detectors "
              "(the YOLO family) changed the cost equation: YOLOv8n, the smallest variant used "
              "here, has roughly three million parameters and is fast enough to run on a CPU at a "
              "reduced frame rate and on a Jetson-class device at video rate.")
    para(doc, "Emergency vehicle preemption in the literature is usually implemented in one of "
              "three ways: (i) acoustic siren detection, (ii) dedicated RF/IR emitters on the "
              "vehicle, and (iii) visual detection. This project follows (iii) because it needs no "
              "cooperation from the vehicle and reuses the same camera that already counts "
              "traffic. The known weakness of (iii) is dataset availability, which is why the "
              "project ships an explicit annotation and training pipeline rather than assuming a "
              "pretrained ambulance class exists.")

    # ---------------- methodology ---------------- #
    h(doc, "3. System Design and Methodology", level=1)
    h(doc, "3.1 Architecture", level=2)
    figure(doc, "docs/arch.png",
           "Figure 1 — Pipeline: four feeds are decoded, downscaled and passed to YOLOv8; the "
           "detections become per-approach counts and priority flags; the controller decides the "
           "green approach; the renderer composes the 2x2 dashboard and the log.", width=6.4)

    para(doc, "The implementation is split so that each concern can be tested on its own:")
    table(doc,
          ["Module", "Responsibility", "Dependencies"],
          [["config.py", "Every tunable value: paths, thresholds, timings, colours", "none"],
           ["detector.py", "Wraps Ultralytics YOLO; converts raw boxes into per-lane counts and "
                           "a priority list; adapts automatically to a custom 4-class model", "torch, ultralytics"],
           ["controller.py", "The decision rules only. Pure logic, clock is injected", "none (unit-tested)"],
           ["smart_intersection.py", "Lane I/O, emergency simulator, renderer, engine loop", "opencv, numpy"],
           ["main.py", "Command-line interface", "argparse"],
           ["web/dashboard.py", "Flask + MJPEG browser dashboard", "flask"]],
          widths=[1.6, 3.5, 1.5])

    h(doc, "3.2 Detection stage", level=2)
    para(doc, "Each frame is downscaled and passed to the model with a confidence threshold of "
              "0.35 and NMS IoU of 0.45. Detection is run every second frame by default "
              "(--detect-every), which halves CPU load with no practical effect on the decision, "
              "because signal states change over seconds, not milliseconds. Two modes exist:")
    bullets(doc, [
        "COCO mode — the shipped yolov8n.pt. Vehicle counting works (car, motorcycle, bus, "
        "truck). COCO contains no ambulance or police class, so emergency events come from the "
        "simulator.",
        "Custom mode — models/best.pt produced by train.py. Classes are ambulance, police, car, "
        "motorcycle. Classification is done by matching class-name keywords, not class indices, "
        "so a model whose class order differs still works.",
    ])
    para(doc, "Mode selection is automatic: if models/best.pt exists it is used, otherwise the "
              "pretrained weights are used. A fresh clone therefore always runs.")

    h(doc, "3.3 Decision logic", level=2)
    para(doc, "The controller is deterministic and side-effect free; time is passed in as an "
              "argument, which is what makes the whole rule set unit-testable without a GPU or a "
              "video. Rules are applied in this order:")
    table(doc,
          ["Order", "Rule", "Behaviour"],
          [["1", "Emergency", "Any approach with >=1 priority vehicle becomes green at once, even "
                              "inside another approach's green time. Two emergencies: the approach "
                              "with more priority vehicles wins; tie goes to lane order (N,S,E,W)."],
           ["2", "Hold", "After the last priority detection the approach stays green for "
                         "emergency_hold seconds (default 6 s) so the convoy can clear."],
           ["3", "Density", "Otherwise the approach with most vehicles gets green, but only if it "
                            "leads the current approach by switch_margin (default 2) — hysteresis "
                            "against one-car flicker."],
           ["4", "Fairness", "Every approach keeps green at least min_green seconds (default 3 s); "
                             "an empty junction keeps the last green rather than going dark."]],
          widths=[0.55, 1.0, 5.0])

    para(doc, "The core of rule 1 and 3, as implemented:")
    code(doc,
         "with_priority = [l for l in lanes if priority[l] > 0]\n"
         "if with_priority:                       # 1. emergency preemption\n"
         "    green = max(with_priority, key=lambda l: priority[l])\n"
         "elif now < hold_until:                  # 2. convoy hold\n"
         "    green = hold_lane\n"
         "else:                                   # 3+4. density with hysteresis\n"
         "    busiest = max(lanes, key=lambda l: counts[l])\n"
         "    if counts[busiest] >= counts[current] + switch_margin:\n"
         "        green = busiest")

    h(doc, "3.4 Rendering and operator interface", level=2)
    para(doc, "The dashboard places the four approaches in a 2x2 grid with a 46-pixel header. Each "
              "tile carries the approach name, vehicle count, a three-lamp signal drawn as vector "
              "graphics, and a status strip. Approaches held at red are dimmed so the green "
              "approach is visually dominant. Bounding boxes are cyan; priority vehicles are "
              "magenta and thicker. When a preemption is active, a magenta banner flashes on the "
              "offending tile and the header names the approach. The same engine can be served to "
              "a browser over MJPEG, with buttons that trigger a simulated ambulance, which makes "
              "the project demonstrable on a machine with no desktop GUI.")

    h(doc, "3.5 Training pipeline", level=2)
    para(doc, "The generic model cannot recognise an ambulance, so the project includes the "
              "complete path from raw video to specialised weights:")
    bullets(doc, [
        "prepare_dataset.py frames — extract stills at a chosen rate from the traffic clips.",
        "prepare_dataset.py prelabel — run the COCO model and write YOLO labels for car, "
        "motorcycle, bus and truck. Human effort is then spent only on ambulance and police boxes.",
        "prepare_dataset.py split — validate image/label pairing, split train/val, report class "
        "balance, and write a data.yaml with absolute paths.",
        "train.py — transfer learning from yolov8n.pt with night-friendly colour augmentation, "
        "automatic device selection, early stopping, and automatic copy of best.pt into models/.",
        "notebooks/train_colab.ipynb — the same job on a free Colab T4 GPU, with per-class AP "
        "reporting so that a good overall mAP cannot hide a weak ambulance class.",
    ], style="List Number")

    # ---------------- results ---------------- #
    h(doc, "4. Results and Discussion", level=1)
    h(doc, "4.1 Working demonstration", level=2)
    figure(doc, "demo/dashboard.png",
           "Figure 2 — Normal operation. The busiest approach (WEST, 7 vehicles) holds green while "
           "the others are red and dimmed. Detection confidence and class are shown per box.")
    figure(doc, "demo/emergency.png",
           "Figure 3 — Emergency preemption. A priority event on NORTH forces NORTH green and "
           "every other approach red within the same frame, with a flashing override banner.")

    h(doc, "4.2 Measured performance", level=2)
    para(doc, "All figures below were measured on a two-vCPU, CPU-only machine (no GPU), four "
              "approaches, input 640 px, detection every second frame:")
    table(doc,
          ["Metric", "Measured value", "Comment"],
          [["Dashboard throughput", "4.5 – 5.1 frames/s", "0.20 s per four-lane step"],
           ["Single YOLOv8n pass", "~72 ms at 640 px", "the detector, not the logic, is the cost"],
           ["Decision computation", "< 1 ms", "pure Python, a few comparisons"],
           ["Latency from detection to signal change", "1 frame (0.20 s)", "verified in demo log"],
           ["Vehicles detected per night frame", "5 – 13 per approach", "pretrained COCO"],
           ["Unit tests passing", "13 / 13", "emergency, hold, density, fairness, hysteresis"]],
          widths=[2.4, 1.7, 2.6])
    para(doc, "Two honest observations. First, the end-to-end rate is limited by inference on CPU; "
              "on a GPU or a Jetson device the same code runs at video rate, and the "
              "--detect-every parameter exists precisely to trade latency for throughput. Second, "
              "the throughput figure is not a claim about emergency response time: the signal "
              "decision is instantaneous once a detection exists, so response time equals detection "
              "latency plus one frame.")

    h(doc, "4.3 Validation of the decision logic", level=2)
    para(doc, "Because the controller takes time as an argument, every rule was tested in "
              "isolation and as a sequence. The suite covers: emergency green overriding the "
              "busiest approach; emergency ignoring an active minimum-green timer; two simultaneous "
              "emergencies; deterministic tie-breaking; the convoy hold expiring into density "
              "control; hysteresis refusing to switch on a one-vehicle lead; exactly one green at "
              "all times; and a regression test for the scheduled-simulation clock. This is the "
              "part of the project that is proven rather than demonstrated.")

    h(doc, "4.4 Limitations", level=2)
    bullets(doc, [
        "Pretrained COCO has no ambulance or police class; real recognition requires the trained "
        "models/best.pt. Until then emergency events are simulated and labelled as simulated on "
        "screen.",
        "The two sample clips contain no emergency vehicles; they are night city traffic.",
        "Two clips feed four approaches, offset so that they do not look identical. A deployment "
        "would use four independent streams.",
        "No physical signal hardware is driven; signal state exists on screen and in the log.",
        "Night-time glare, reflections and low contrast reduce confidence; the training "
        "configuration compensates with strong value and saturation augmentation, but a "
        "night-specific dataset is the better fix.",
    ])

    # ---------------- conclusion ---------------- #
    h(doc, "5. Conclusion and Future Work", level=1)
    para(doc, "A four-approach signal controller that prioritises emergency vehicles has been "
              "designed, implemented, demonstrated and tested. The decision layer is proven by "
              "unit tests; the perception layer is functional with pretrained weights and is "
              "designed to be specialised with a documented training pipeline; and the operator "
              "interface exists in both desktop and browser forms. The remaining gap is "
              "annotated ambulance imagery and GPU time, not software.")
    table(doc,
          ["Phase", "Future work"],
          [["1", "Annotate Indian ambulance (108 / EMRI) and police vehicles, train and evaluate "
                 "per-class AP50; integrate the resulting best.pt"],
           ["2", "Deploy on Jetson Nano / Raspberry Pi inside the signal cabinet, with an ESP32 or "
                 "relay board driving the physical lamps over serial or MQTT"],
           ["3", "Fuse siren detection (acoustic) with vision to catch emergency vehicles that are "
                 "occluded or approaching from outside the field of view"],
           ["4", "Coordinate several junctions to create a green corridor along an ambulance route"],
           ["5", "Log historical density per approach and learn time-of-day green splits instead of "
                 "reacting frame by frame"]],
          widths=[0.7, 5.8])

    h(doc, "References", level=1)
    refs = [
        "J. Redmon, S. Divvala, R. Girshick, A. Farhadi, \"You Only Look Once: Unified, Real-Time "
        "Object Detection\", CVPR 2016.",
        "G. Jocher, A. Chaurasia, J. Qiu, \"Ultralytics YOLOv8\", 2023. [Online]. "
        "Available: https://github.com/ultralytics/ultralytics",
        "T.-Y. Lin et al., \"Microsoft COCO: Common Objects in Context\", ECCV 2014.",
        "R. K. Megalingam et al., \"Smart traffic signal control for emergency vehicles using "
        "image processing\", IEEE conference proceedings.",
        "OpenCV documentation, https://docs.opencv.org",
        "Indian Roads Congress / MoRTH guidelines on emergency vehicle priority at signalised "
        "intersections.",
    ]
    for i, r in enumerate(refs, 1):
        p = doc.add_paragraph(f"[{i}]  {r}")
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.left_indent = Inches(0.35)
        p.paragraph_format.first_line_indent = Inches(-0.35)
        p.runs[0].font.size = Pt(9.5)

    h(doc, "Appendix A — Reproducing the results", level=1)
    code(doc,
         "git clone https://github.com/Devesh-tiw/Priority_Traffic_router.git\n"
         "cd Priority_Traffic_router\n"
         "pip install -r requirements.txt\n\n"
         "# live dashboard (hotkeys 1-4 simulate an ambulance)\n"
         "python main.py\n\n"
         "# reproduce the demo video and the decision log\n"
         "python main.py --headless --save demo/demo.mp4 --frames 500 \\\n"
         "               --simulate NORTH:3:7 --simulate EAST:12:7 \\\n"
         "               --log demo/decision_log.csv\n\n"
         "# browser dashboard\n"
         "python web/dashboard.py\n\n"
         "# test the decision logic\n"
         "python -m unittest discover -s tests -v")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print(f"wrote {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    build()
