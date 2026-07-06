<p align="center">
  <h1 align="center">[ECCV 2026] Scal3R: Learning Efficient Multi-Relative Pose Query for Scalable Online 3D Reconstruction</h1>
  <p align="center">
    <a href="https://linjohnss.github.io/"><strong>Chin-Yang Lin</strong></a>
    ·
    <a href="https://tw.linkedin.com/in/yang-che-sun-6b9341282"><strong>Yang-Che Sun</strong></a>
    ·
    <a href="https://sunset1995.github.io/"><strong>Cheng Sun</strong></a>
    ·
    <a href="https://fuenyang1127.github.io/"><strong>Fu-En Yang</strong></a>
    <br>
    <a href="https://minhungchen.netlify.app/"><strong>Min-Hung Chen</strong></a>
    ·
    <a href="https://sites.google.com/site/yylinweb/"><strong>Yen-Yu Lin</strong></a>
    ·
    <a href="https://walonchiu.github.io/"><strong>Wei-Chen Chiu</strong></a>
    ·
    <a href="https://yulunalexliu.github.io/"><strong>Yu-Lun Liu</strong></a>
  </p>
  <h3 align="center"><a href="https://linjohnss.github.io/scal3r/">Project Page</a> | <a href="https://linjohnss.github.io/scal3r/">Paper</a> | <a href="https://huggingface.co/nvidia/scal3r">🤗 Hugging Face</a></h3>
  <div align="center"></div>
</p>

<p align="center">
  <a href="">
    <img src="./assets/teaser.gif" alt="Teaser" width="100%">
  </a>
</p>

<div align="center">
TL;DR: Scalable online 3D reconstruction on kilometer-scale sequences, with only ~1% extra parameters on a frozen backbone trained in 8 hours on a single GPU.
</div>
<br>

## Overview

Scal3R reformulates online 3D reconstruction as **multi-reference relative pose querying**. A small set of learnable pose query tokens (~1% of parameters) is injected into a completely frozen backbone via **asymmetric attention**, and the predicted relative constraints are aggregated by online **pose-graph optimization** with keyframe selection and loop closure, suppressing long-range drift while fully preserving the backbone's pointmap quality.

## Getting Started

We provide Scal3R implementations on two online 3D reconstruction backbones. Please refer to the README in each subdirectory for installation, inference, evaluation, and training:

- [**scal3r_cut3r**](scal3r_cut3r/README.md) — Scal3R on [CUT3R](https://github.com/CUT3R/CUT3R) (persistent state model)
- [**scal3r_stream3r**](scal3r_stream3r/README.md) — Scal3R on [STream3R](https://github.com/NIRVANALAN/STream3R) (causal Transformer model)

## Citation

If you find our work useful, please cite:

```bibtex
@inproceedings{lin2026scal3r,
  title={Scal3R: Learning Efficient Multi-Relative Pose Query for Scalable Online 3D Reconstruction},
  author={Lin, Chin-Yang and Sun, Yang-Che and Sun, Cheng and Yang, Fu-En and Chen, Min-Hung and Lin, Yen-Yu and Chiu, Wei-Chen and Liu, Yu-Lun},
  booktitle={ECCV},
  year={2026}
}
```
