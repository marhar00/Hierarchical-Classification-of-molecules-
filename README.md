# Hierarchical Molecular Classification with Deep MLP

This repository contains a solution for the **EnsembleAI Hackathon 2026 – ChEBI ontology prediction task**. The goal is to predict **hierarchical chemical classes** for molecules based on their **SMILES representations**.

The project implements a **deep multilayer perceptron (MLP)** trained on **molecular fingerprints**, with **hierarchical constraints** enforced during training and prediction.

---

# Problem Description

The task is a **multi-label hierarchical classification problem**.

Each molecule may belong to multiple chemical classes from the **ChEBI ontology**. These classes form a **Directed Acyclic Graph (DAG)**.

Example hierarchy:

```
Organic molecule
   └── Cyclic compound
          └── Polycyclic compound
```

If a molecule belongs to a **child class**, it must also belong to all its **parent classes**.

The model therefore must:

1. Predict **500 binary class labels**
2. Respect the **ontology hierarchy**
3. Handle **highly imbalanced data**

The evaluation metric is **macro-averaged F1 score** across all classes.

---

# Method Overview

The solution follows this pipeline:

```
SMILES
   ↓
Molecular graph
   ↓
Fingerprints (ECFP + MACCS + Topological Torsion)
   ↓
Feature concatenation
   ↓
Deep MLP classifier
   ↓
Focal loss + Hierarchical loss
   ↓
Threshold optimization
   ↓
Hierarchy-consistent predictions
```

---

# Feature Engineering

Molecular features are extracted using **scikit-fingerprints**.

Three fingerprint types are concatenated:

### 1. ECFP (Extended Connectivity Fingerprint)

Captures circular neighborhoods around atoms.

```
dimension = 2048
radius = 2
```

---

### 2. MACCS keys

Expert-defined substructure fingerprints.

```
dimension ≈ 167
```

---

### 3. Topological Torsion Fingerprints

Captures atom sequences and topological patterns.

```
dimension = 2048
```

---

### Final feature vector

```
X = [ECFP | MACCS | Torsion]
```

Total feature dimension:

```
≈ 4200
```

---

# Model Architecture

A **deep MLP** is used as the classifier.

```
Input layer (~4200)

↓ Linear
↓ BatchNorm
↓ GELU
↓ Dropout

↓ Linear
↓ BatchNorm
↓ GELU
↓ Dropout

↓ Linear
↓ BatchNorm
↓ GELU
↓ Dropout

↓ Linear (500 outputs)
```

Example configuration:

```
Hidden layers: [2024, 1012, 512]
Dropout: 0.3
Activation: GELU
```

---

# Handling Class Imbalance

The dataset is highly imbalanced (many rare classes).

We use **Focal Loss**:

```
FL = α (1 − pt)^γ BCE
```

Where:

- γ = 2
- pos_weight computed from class frequencies

This focuses training on **hard examples**.

---

# Hierarchical Consistency

The class hierarchy is parsed from:

```
chebi_classes.obo
```

A **parent mask matrix** is constructed:

```
parent_mask[i,j] = 1
if class j is parent of class i
```

---

## Hierarchical Loss

To prevent invalid predictions:

```
p(child) ≤ p(parent)
```

Violation penalty:

```
max(0, logit_child − logit_parent)
```

Final training loss:

```
Total Loss = FocalLoss + λ * HierarchicalLoss
```

where

```
λ = 0.7
```

---

# Training Details

| Parameter | Value |
|-----------|------|
| Optimizer | AdamW |
| Learning Rate | 1e-3 |
| Weight Decay | 1e-4 |
| Batch Size | 256 |
| Epochs | 40 |
| Scheduler | CosineAnnealingWarmRestarts |

Gradient clipping is used:

```
max_norm = 5
```

---

# Results (Validation)

Typical training results:

```
F1-micro ≈ 0.906
F1-macro ≈ 0.800
Hierarchy violation rate ≈ 0.5%
```

Hierarchy violations decrease during training thanks to the hierarchical loss.

Maximal training results:

```
F1-micro ≈ 0.93
F1-macro ≈ 0.84
Hierarchy violation rate ≈ 0.3%
```

---

# Threshold Optimization

Since each class has different prevalence, a **separate threshold per class** is optimized.

Procedure:

1. Compute predictions on validation set
2. Use **precision-recall curve**
3. Choose threshold maximizing **F1 score**

```
threshold_i = argmax F1(class_i)
```

---

# Hierarchy-aware Prediction

After thresholding, predictions are corrected:

```
if child = 1
   parent = max(parent, child)
```

This guarantees **ontology consistency**.

---

# Project Structure

```
.
├── README.md
├── chebi_dataset_train.parquet
├── chebi_dataset_test_empty.parquet
├── chebi_classes.obo
├── chebi_class_definitions.csv
│
├── outputs/
│   ├── training_curves_mlp_fp.png
│   └── chebi_submission_mlp_fp.parquet
```

---


# Libraries Used

Main libraries:

```
PyTorch
NumPy
Pandas
scikit-learn
scikit-fingerprints
RDKit
matplotlib
```

---

# Possible Improvements

Future improvements could include:

- Graph Neural Networks (PyTorch Geometric)
- SMILES transformers (ChemBERTa)
- Pretrained molecular embeddings
- Graph transformers
- Using LightGBM for ensembling - in another notebook i created a model, but couldn't run it locally - my laptop couldn't handle it. In the future i will try to train it on a VM.



---

# Author

Marysia Harbaty

Project created for **EnsembleAI Hackathon 2026**.