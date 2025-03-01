# synthlifegaming
Using AI to bring gaming to synth life 2D and 3D platforms.
SynthLifeGaming
Using AI to bring gaming to synthetic life on 2D and 3D platforms. This project features a 3D simulation where AI-controlled central spheres evade shots from player-controlled bots, leveraging reinforcement learning to enhance evasion strategies.
Description
SynthLifeGaming is an interactive simulation built with Python, VPython, and PyTorch. It runs in two modes: a training mode where the AI learns to evade shots from multiple bot-controlled spheres, and a real-play mode where users can control a green sphere to shoot at an AI-driven central sphere. The AI uses a neural network (EvasionNet) to decide movement, shooting, and teleportation based on environmental inputs like shot positions and boundary distances.
Installation
To run the simulation, install the following dependencies:
Prerequisites
Python: Version 3.7 or later (python --version to check)

A graphical environment (e.g., desktop or Jupyter notebook with VPython support)

Dependencies
Package

Command

Notes

VPython

pip install vpython

For 3D graphics

PyTorch

pip install torch
 or see 
PyTorch Installation
GPU support recommended

NumPy

pip install numpy

Usually included with VPython

Example installation:
bash

pip install vpython torch numpy

Usage
Run the Simulation:
Save the code as main.py (or your preferred name).

Execute it with: python main.py.

A 3D window will open showing the simulation.

Modes:
Training Mode: Default mode. Watch multiple central spheres (orange) evade shots from corner and center bots (blue/green). Press r to switch to real-play.

Real-Play Mode: Control the green sphere with arrow keys and shoot with the spacebar. Press q to return to training.

Controls (Real-Play Mode):
Arrow Keys: Move the green sphere (left, right, up, down).

Spacebar: Shoot at the central sphere.

Y / I: Zoom in/out.

Simulation Details:
The AI trains every 72 seconds (if 1000+ data points are collected) and saves the model to phase2_evasion_model98888888888888.pth.

Use the mouse to rotate, zoom, or pan the 3D view.

How It Works
Environment: A 50x50 platform with a central sphere (AI-controlled), player spheres (bots or user-controlled), and projectiles.

AI Model: EvasionNet processes 29 inputs (e.g., positions, velocities, distances) to output movement (dx, dz), a shoot decision, and teleportation targets (tx, tz).

Training: Uses reinforcement learning with epsilon-greedy exploration, AdamW optimizer, and a custom loss (MSE + BCE).

Contributing
Contributions are welcome! Feel free to:
Open an issue for bugs or suggestions.

Submit a pull request with improvements.

Please ensure your code follows Python conventions and includes comments for clarity.
License
This project is licensed under the [MIT] - see the LICENSE file for details.
This project was created with help from Grok 3 from grok.com enjoy!


