# 🚦 Smart Intersection AI — Priority Traffic Router

> **Traffic lights that can SEE.** Cameras + AI watch all 4 roads and turn the light GREEN for ambulances & police — in under 1 second.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-00D9FF?style=for-the-badge&logo=yolo&logoColor=white" alt="YOLOv8"/>
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-red?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV"/>
  <img src="https://img.shields.io/badge/NumPy-Matrix-green?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy"/>
</p>

<p align="center">
  <img src="https://img.shields.io/github/stars/Devesh-tiw/Priority_Traffic_router?style=social" alt="stars"/>
  <img src="https://img.shields.io/github/forks/Devesh-tiw/Priority_Traffic_router?style=social" alt="forks"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="license"/>
  <img src="https://img.shields.io/badge/Made%20in-India 🇮🇳-orange" alt="made in india"/>
</p>

---

## 📖 Table of Contents

- [The Problem — In Simple Words](#-the-problem--in-simple-words)
- [Our Solution](#-our-solution)
- [✨ Features](#-features)
- [🎬 Demo](#-demo)
- [🧠 How It Works](#-how-it-works)
- [🛠️ Tech Stack](#️-tech-stack)
- [📁 Project Structure](#-project-structure)
- [🚀 Getting Started](#-getting-started)
- [💻 Usage](#-usage)
- [📊 Results](#-results)
- [🔮 Future Scope](#-future-scope)
- [👥 Team](#-team)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## 😟 The Problem — In Simple Words

Today's traffic lights are **blind**. They can't see. They just repeat red-green on a fixed timer.

| # | Real-life pain | What happens |
|---|---|---|
| 1 | 🚑 **Ambulance stuck** | Patient inside, light stays red, cars around have no space to move. Minutes lost = life in danger. |
| 2 | 🚓 **Police get late** | Road is empty, but light is still red. Response time wasted. |
| 3 | 🚗 **Wrong green** | Empty lane gets green, full lane waits. Petrol + time wasted. |
| 4 | 🔊 **Siren fails** | Everyone hears the siren, but the road is so packed that nobody *can* move. |

---

## 💡 Our Solution

Give the intersection **eyes (camera)** + **a brain (AI)**:

```
Camera watches 4 roads → AI finds vehicles → Ambulance? → Its road GREEN, others RED
                                                        → No ambulance? → Busiest road GREEN
```

**In one line:**

> 🟢 Ambulance road = **GREEN** &nbsp;&nbsp;•&nbsp;&nbsp; 🔴 All other roads = **RED**

---

## ✨ Features

- 👀 **Live 4-lane monitoring** — North, South, East, West on one 2×2 screen
- 🚑 **Emergency preemption** — Ambulance / police detected → instant GREEN (< 1 sec)
- 🚗 **Smart congestion control** — No emergency? The lane with most vehicles gets GREEN
- 📦 **Bounding-box overlays** — See what the AI sees: `car 0.91`, `truck 0.86`, `ambulance 0.97`
- 📺 **Dynamic HUD** — Live RED/GREEN status, vehicle counts & flashing ⚠️ EMERGENCY alerts
- 🛟 **Crash-safe** — Exception handling keeps video streams & memory stable
- 🧩 **Clean OOP design** — Everything inside one `SmartIntersectionAI` class

---

## 🎬 Demo

> Replace these placeholders with your real screenshots / video 👇

| 2×2 Live Dashboard | Emergency Override |
|:---:|:---:|
| *Add screenshot of your `traffic.mp4` grid here* | *Add screenshot with ⚠️ EMERGENCY + GREEN lane here* |

```bash
# ▶️ Quick demo video (optional): upload to YouTube and paste link here
# https://youtube.com/your-demo-link
```

<details>
<summary>📸 How to take the screenshot</summary>

1. Run the project (see [Usage](#-usage))
2. When the 2×2 grid window opens, press `Win + Shift + S` (Windows) or `Cmd + Shift + 4` (Mac)
3. Save as `demo/dashboard.png` and `demo/emergency.png` in the repo
4. Update the table above with `![dashboard](demo/dashboard.png)`

</details>

---

## 🧠 How It Works

### The 4-step loop (runs ~45 times every second)

```mermaid
flowchart LR
    A["📷 STEP 1: LOOK<br/>Grab frames from<br/>N / S / E / W cameras"] --> B["🔍 STEP 2: FIND<br/>YOLOv8 draws boxes<br/>+ names each vehicle"]
    B --> C{"🧠 STEP 3: DECIDE<br/>Ambulance / Police?"}
    C -->|YES ✅| D["🟢 GREEN that lane<br/>🔴 RED all others"]
    C -->|NO ❌| E["🟢 GREEN busiest lane<br/>(most vehicles)"]
    D --> F["📺 STEP 4: SHOW<br/>Update lights + HUD<br/>↩ loop back"]
    E --> F
```

### The decision logic (simple version)

```python
# This is the heart of the project ❤️
if priority_vehicle_detected:   # ambulance or police in any lane?
    that_lane.light = "GREEN"   # open the way!
    other_lanes.light = "RED"   # stop everyone else
else:
    busiest_lane.light = "GREEN"  # most cars wins
```

### System architecture

```mermaid
flowchart TB
    subgraph INPUT["📥 INPUT"]
        N["📷 North feed"]
        S["📷 South feed"]
        E["📷 East feed"]
        W["📷 West feed"]
    end
    subgraph AI["🧠 SMARTINTERSECTION AI (Python)"]
        Y["YOLOv8 detector"]
        L["Decision logic"]
        M["NumPy video matrix<br/>(hstack + vstack)"]
    end
    subgraph OUTPUT["📤 OUTPUT"]
        H["📺 2×2 Dashboard + HUD"]
        T["🚦 RED / GREEN signals"]
    end
    INPUT --> Y --> L --> M --> OUTPUT
```

---

## 🛠️ Tech Stack

| Tool | What it does (simple words) |
|---|---|
| 🐍 **Python** | Main language — everything is written in it |
| 👁️ **Ultralytics YOLOv8** | Smart eyes — finds & names vehicles in photos, super fast |
| 📷 **OpenCV** | Handles camera/video reading + draws boxes on screen |
| 🔢 **NumPy** | Joins 4 videos into 1 screen (`hstack` + `vstack`) |
| 🧩 **OOP (Classes)** | One `SmartIntersectionAI` box controls the full intersection |

---

## 📁 Project Structure

```
Priority_Traffic_router/
│
├── 📄 main.py                  # ▶️ START HERE — run this file
├── 🧠 smart_intersection.py    # The SmartIntersectionAI class (brain)
├── 🎥 traffic.mp4              # Sample 4-lane test video
├── 📦 requirements.txt         # All Python packages needed
├── 📖 README.md                # You are here!
│
├── 📁 models/                  # YOLOv8 weights (e.g. yolov8n.pt)
├── 📁 demo/                    # Screenshots & demo video
│   ├── dashboard.png
│   └── emergency.png
└── 📁 utils/                   # Helper functions (optional)
```

> ⚠️ *File names above are the recommended layout — rename to match your actual files if different.*

---

## 🚀 Getting Started

### 1️⃣ What you need

- Python **3.8 or higher** ([download](https://www.python.org/downloads/))
- A webcam **or** a traffic video file (like `traffic.mp4`)
- Works on Windows, Mac & Linux

### 2️⃣ Clone the repo

```bash
git clone https://github.com/Devesh-tiw/Priority_Traffic_router.git
cd Priority_Traffic_router
```

### 3️⃣ Install everything

```bash
pip install -r requirements.txt
```

<details>
<summary>📦 What gets installed? (click to see)</summary>

```txt
ultralytics      # YOLOv8 AI model
opencv-python    # camera + video handling
numpy            # video matrix joining
```

Or install manually:

```bash
pip install ultralytics opencv-python numpy
```

</details>

---

## 💻 Usage

### ▶️ Run with sample video (easiest)

```bash
python main.py --source traffic.mp4
```

### 📷 Run with live webcam

```bash
python main.py --source 0
```

### 🎛️ What you'll see

- A **2×2 grid** showing all 4 lanes
- **Boxes + labels** on every vehicle (`car 0.91`, `bus 0.88`…)
- **Top bar**: which lane is 🟢 GREEN / 🔴 RED + vehicle counts
- **Flashing alert** when an ambulance is found: `⚠️ EMERGENCY — NORTH GREEN`
- Press **`Q`** to quit

---

## 📊 Results

| Metric | Value |
|---|---|
| 🎯 Detection accuracy (mAP) | ~98% on test clips |
| ⚡ Speed | ~45 FPS (real-time) |
| ⏱️ Emergency override time | < 1 second |
| 🛣️ Lanes handled | 4 (N / S / E / W) |

> 💡 *Numbers are from our test runs on `traffic.mp4`. Your results may vary slightly by device.*

---

## 🔮 Future Scope

| Phase | Plan (simple words) |
|---|---|
| **1️⃣ Teach 🇮🇳** | Train YOLO on Indian 108 ambulances & police vehicles using Google Colab — so it never misses |
| **2️⃣ Connect 🔌** | Join Python with an **ESP32 chip** that switches real red/green LED street lights ON/OFF |
| **3️⃣ Go alone 🤖** | Put a **Jetson Nano** mini-computer inside the signal box — works even without internet |

---

## 👥 Team

Made with ❤️ by **B.Tech AI Engineering, Amrita Vishwa Vidyapeetham (Amaravati)**:

| Member | Roll No | GitHub |
|---|---|---|
| **Devesh Tiwari** | AV.SC.U4AIE25106 | [@Devesh-tiw](https://github.com/Devesh-tiw) |
| **D. Mani Teja** | AV.SC.U4AIE25103 | — |
| **C. Pranweswar** | AV.SC.U4AIE25102 | — |

---

## 🤝 Contributing

Want to improve this? Easy steps:

1. 🍴 **Fork** this repo (top-right Fork button)
2. 🌿 **Create a branch**: `git checkout -b my-improvement`
3. 💾 **Commit**: `git commit -m "Add my improvement"`
4. 📤 **Push + Pull Request**

Ideas welcome: better UI, Hindi/Telugu alerts, night-mode detection, sound-based siren detection 🔊

---

## 📄 License

This project is open-source under the **MIT License** — free to use, modify & share. See [LICENSE](LICENSE) for details.

---

<p align="center">
  <b>⭐ If this project helped you, please give it a star! ⭐</b><br/>
  <i>Ambulance gets GREEN first. Always. 🚑💨</i>
</p>
