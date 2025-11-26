# Dreamer Bot

This transformer-based agent uses screen and audio input to **predict and execute actions**. It operates in two modes: imitation learning and autonomous control.

---

## Prerequisites

* **Python 3.8+**
* **CUDA-capable GPU (recommended)**

---

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

**Note:** If using an NVIDIA GPU, install the CUDA version of PyTorch manually *before* running the requirements file to ensure hardware acceleration.

---

## Usage

Run the main script:

```bash
python main.py
```

---

## Calibration

1. Hover mouse over the **top-left corner** of the capture area and press **`k`**.
2. Hover mouse over the **bottom-right corner** and press **`k`** again.

---

## Modes

### **[1] PLAY MODE (Autonomous)**

The agent controls mouse and keyboard inputs based on:

* its training
* current observations

### **[2] WATCH MODE (Imitation)**

Records:

* screen
* audio
* user inputs

…for training, without interfering with controls.

---

## Controls

The agent tracks:

* **Keys:** `W, A, S, D, Space, Shift, Ctrl, Q, E`
* **Mouse:** left/right click, scroll, movement

---

## Stopping

Press **F9** to save the checkpoint and exit safely.

---

## Configuration

### **Checkpoints**

* Weights save to `ckpts_predictive_action` every **15 minutes** or upon exit.
* The script automatically loads the **latest checkpoint**.

### **Adjustable Variables (in script header)**

* **IMG_H, IMG_W** – Input tensor resolution (default: `128x256`)
* **MOUSE_SPEED** – Mouse movement distance per step
* **INPUT_DURATION** – Key press duration during autonomous play
* **DREAM_HORIZON** – Internal prediction steps
* **KEY_LIST** – Customize tracked keys (e.g. `['z', 'x']`)

  * Ensure names match the `keyboard` library.

---

## Notes

* **Failsafe:** `pyautogui` failsafe is disabled. Use **F9** to exit.

---
