# PriorFormer: A Transformer for Real-time Monocular 3D Human Pose Estimation with Versatile Geometric Priors

Official implementation of the paper:  
**PriorFormer: A Transformer for Real-time Monocular 3D Human Pose Estimation with Versatile Geometric Priors**  
*Mohamed Adjel, Vincent Bonnet*  
Accepted at **IEEE-RAS International Conference on Humanoid Robots (Humanoids 2025)**

---

## Overview
This repository provides code and pretrained models for **PriorFormer**, a lightweight Transformer-based lifter that maps short sequences of **2D human joint positions** to **3D poses** using a single monocular camera.  

Unlike previous approaches, PriorFormer:
- Supports **optional geometric priors** (camera intrinsics and bone segment lengths),
- Operates in both **calibrated and uncalibrated settings**,
- Uses a **masking mechanism** to remain robust when priors are missing,
- Runs in **real time** on CPU and GPU, making it suitable for **edge devices**.

Key features:
- Fast: 380 µs on GPU / 1.8 ms on CPU  
- Lightweight: <0.6M parameters  
- Accurate: ~36 mm MPJPE on synthetic AMASS-based benchmarks  
- Versatile: Works with or without geometric priors  

---

## Paper
If you use this repository, please cite our paper:

```bibtex
@inproceedings{adjel2025priorformer,
  title={PriorFormer: A Transformer for Real-time Monocular 3D Human Pose Estimation with Versatile Geometric Priors},
  author={Adjel, Mohamed and Bonnet, Vincent},
  booktitle={IEEE-RAS International Conference on Humanoid Robots (Humanoids)},
  year={2025}
}
