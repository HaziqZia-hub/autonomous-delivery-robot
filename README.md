# Autonomous Delivery Robot (UGV)

An autonomous unmanned ground vehicle built around a Raspberry Pi 5 SBC. The system fuses incoming data from a 2D LiDAR, an IMU, and wheel encoders to implement a live SLAM model for real-time navigation. The localization pipeline utilizes a tinyslam methodology for rapid pose estimation, while mapping is handled via Iterative Closest Point (ICP) scan matching.

---

## 🚀 Key Features
* Real-Time SLAM: Live map generation and precise positioning using sensor fusion.
* Dead Reckoning Localisation: Fuses high-resolution wheel encoders and IMU data to maintain accurate pose estimations between LiDAR scans.
* Custom Power Distribution: Dedicated power rails designed to safely handle heavy current draws from drive motors while delivering clean, regulated power to the Raspberry Pi 5.

---

## 🛠️ Hardware & Circuit Architecture
The system architecture spans low-level analog components up to high-level single-board computing.

* Compute: Raspberry Pi 5 SBC (Running headless Linux environment via SSH/VNC)
* Sensors: 2D LiDAR, 6-DoF IMU, Optical Wheel Encoders
* Power System: Custom battery configuration with integrated BMS and high-efficiency buck converters for logic circuitry.
* Drive System: Discrete transistor-driven/H-bridge motor control paths.

### Circuit Schematic & System Architecture
>>[circuit.pdf](https://github.com/user-attachments/files/28068148/circuit.pdf)

---

## 💻 Software Stack & Algorithms
* Language: Python
* Localization: Tinyslam core logic optimized for low-overhead embedded execution.
* Scan Matching: Iterative Closest Point (ICP) matching for mapping consistency.
* Motor Control: Custom PWM generation and feedback loop logic for speed regulation.

---

## 📹 Performance & Demos
Here is the robot operating in real-time, performing live localization and map adjustments:

>>https://github.com/user-attachments/assets/693644e9-d46e-4a47-b8a2-cbfa901763c1
>>https://github.com/user-attachments/assets/a8831505-0ee9-4f98-a8be-62de385d34a7

---

## 📁 Repository Structure
* `/firmware`: Embedded control loops, sensor parsing scripts, and navigation logic.
* `/hardware`: Component layouts, Fritzing schematics.
* `/documentation`: Project report, presentation slides & High-resolution PDFs of full schematics
