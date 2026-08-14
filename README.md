# Passive Aerodynamic Optimization for Low-Speed Airfoils

Welcome! This repository contains the code and simulation pipeline for my research on **geometry-only, passive control of Laminar Separation Bubbles (LSBs)** on low-Reynolds-number airfoils.

Instead of adding physical turbulator strips, tape, or powered air actuators, this project uses mathematical shape tuning to reshape the airfoil surface, smoothing out pressure changes and suppressing separation naturally.

---

## The Problem: What is a Laminar Separation Bubble?

When small aircraft, high-altitude drones, or micro air vehicles fly at low speeds, the air moving over the wing starts off smooth and laminar. However, as it passes the thickest point of the wing, it hits an adverse pressure gradient.

Because smooth laminar airflow lacks energy near the wall, it detaches from the wing surface, forms a swirling pocket of recirculating air called a **Laminar Separation Bubble (LSB)**, and eventually transitions into turbulent air before reattaching.

These separation bubbles cause significant performance issues:

* Unwanted pressure drag
* Reduced aerodynamic efficiency ($L/D$)
* Higher risks of premature stall

---

## The Solution: Passive Shape Tuning

Traditional fixes usually involve sticking physical devices onto the wing, such as zig-zag tape or turbulator wires. While these force early transition to prevent separation, they also add constant friction drag—even when you don't need them.

**My approach:** Use computational fluid dynamics (CFD) paired with gradient-based optimization to subtly morph the airfoil's upper surface contour. By easing the pressure gradients along the suction side, we can shorten or eliminate the bubble purely through geometry.

---

## How it Works

The pipeline connects three core components:

1. **Shape Representation (CST):** The airfoil is parameterized using Class-Shape Transformation (CST) with 5th-degree Bernstein polynomials. This gives us 12 flexible variables to smoothly control the curve.
2. **Flow Simulation (SU2):** Flow fields are resolved using the open-source **SU2** solver, combined with the transition-aware $\gamma\text{-}Re_{\theta}$ Langtry–Menter model to track exactly where the air separates and reattaches.
3. **Optimizer Driver (MMA):** Svanberg's Method of Moving Asymptotes calculates how tiny shape adjustments impact drag, steering the geometry toward an optimal shape while keeping lift and structural thickness within required safety bounds.

---

## Key Results

Under test conditions ($Re = 1.0 \times 10^5$, $\alpha = 4.0^\circ$), the 15-iteration optimization sequence delivered clear performance gains:

* **40.8% Drag Reduction:** Profile drag ($C_D$) dropped from **0.3333** down to **0.1972**.
* **31.3% Efficiency Gain:** The lift-to-drag ratio ($L/D$) increased from **3.945** to **5.179**.
* **Constraints Satisfied:** Maintained target lift ($C_L \ge 1.0$) and structural thickness ($t/c \ge 9\%$).

---

## Code Overview

* `cst_geometry.py` — Handles airfoil geometry parameterization and surface mesh generation.
* `optimize_mma.py` — Runs the gradient evaluations and optimization step updates.
* `literature_validator.py` — Validates solver performance against experimental datasets (Eppler 387 and SD7003 airfoils).
* `su2_config/` — Contains solver configuration parameters and grid setup files.
* `scripts/` — Execution scripts for running the optimization loop and generating plots.

---

## Quick Start

### Requirements

* Python 3.9+
* SU2 CFD Suite (installed and added to system `PATH`)
* Standard Python libraries: `numpy`, `scipy`, `matplotlib`

### Running the Pipeline

Clone the repository and launch the main run script:

```bash
git clone https://github.com/eshantsonhar/Airfoil-Generator-Model.git
cd Airfoil-Generator-Model
python scripts/run_pipeline.py

```

To view the pressure distributions ($C_p$) and optimization history:

```bash
python scripts/plot_results.py

```

---
