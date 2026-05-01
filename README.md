# Topology-Guided Feature Learning for OCTA Vessel Segmentation (TAG-Net)

## Overview
This project presents **Topology-Guided Adaptive Fusion Network (TAG-Net)**, a deep learning framework for retinal vessel segmentation in Optical Coherence Tomography Angiography (OCTA). This work integrates **topological priors directly into feature learning**, improving vessel connectivity and structural consistency.

---

## Results / Visualizations

### Model Architecture
![TAG-Net Architecture](assets/Architecture.png)

<!--### Qualitative Results
![Results](images/results.png)

> Place your images inside an `images/` folder in your repository.
-->

---

## Key Contributions
- Introduces **feature-level topology integration** (not just supervision)
- Proposes a **topology prediction branch** for soft vessel skeletons
- Develops a **topology-guided fusion module**
- Adds **confidence-based refinement** for thin vessel segmentation
- Improves **connectivity (clDice)** and thin vessel detection across datasets

---

## Method Overview

The proposed TAG-Net consists of three main components:

1. **Topology Prediction Branch**
   - Learns soft vessel skeleton representations

2. **Topology-Guided Feature Fusion**
   - Injects structural information into segmentation features

3. **Refinement Module**
   - Enhances thin vessel segmentation using confidence-based reweighting

---

## Installation

```bash
git clone https://github.com/yourusername/tag-net-octa.git
cd tag-net-octa
pip install -r requirements.txt
```

---

## Train Model
python src/train_full_tagnet_model.py --name TrainedModelFolder  --net full_model --backbone unet  --epochs 200 --batch_size 16 --num_classes 3


---
## Evaluate Model
python src/test_topolgy.py --net full_model --backbone unet --name TrainedModel --evaluate --save_results --save_results_dir result/TrainedModelFolder/Full_Model





