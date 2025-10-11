# ML Challenge 2025 — Smart Product Pricing  
**Team Name:** Synaptech  
**Team Members:** Viraj Kumar, Sarvagya Dwivedi, Amon Harsh, Ansh Arya  
**Submission Date:** 11 October 2025  

---

## 1. Executive Summary
We developed a **hybrid multimodal model** that integrates **textual** and **visual** features to predict product prices.  
Our approach combines:
- **TF–IDF-based textual embeddings** to capture product semantics and numerical cues.  
- **CNN-based image embeddings (ResNet-50)** to represent visual product characteristics.  
- A **LightGBM regressor** for final price prediction.

This architecture leverages both linguistic and visual modalities, achieving robust performance and interpretability with efficient computation.

---

## 2. Methodology Overview

### 2.1 Problem Analysis
**Objective:** Predict the product price using catalog text (`catalog_content`) and product image (`image_link`).  

**Key Observations:**
- Textual data contains strong numeric and descriptive signals such as *IPQ, quantity, and model codes*.  
- Preserving rare tokens (e.g., brand/model IDs) improved model accuracy.  
- Visual patterns like product type and packaging had measurable impact on price distribution.

---

### 2.2 Solution Strategy
**Approach Type:** Hybrid (Text + Image)  
**Core Innovation:**  
We combined TF–IDF-based text representations with deep image embeddings extracted using **ResNet-50**, creating a unified multimodal feature space.  
A **LightGBM model** trained on this fusion achieved strong generalization across diverse product categories.  

**Evaluation Metric:** SMAPE (Symmetric Mean Absolute Percentage Error)

---

## 3. Model Architecture

### 3.1 Overview
catalog_content → preprocessing → TF–IDF → text embeddings
image_link → download → transform → ResNet-50 → image embeddings
text + image embeddings → concatenation → LightGBM → price prediction


---

### 3.2 Model Components

#### 📝 Text Processing Pipeline
- **Preprocessing:**  
  - Lowercasing  
  - Tokenization (retain numbers and hyphens)  
  - Minimal stopword removal  
  - URL and unwanted numeric removal  

- **Feature Extraction:** TF–IDF vectorizer  
  - `max_features = 30,000`  
  - `ngram_range = (1, 2)`  

- **Output:** Dense TF–IDF embeddings  

---

#### 🖼️ Image Processing Pipeline
- **Preprocessing:**  
  - Resize → (224 × 224)  
  - Normalize → Tensor format  

- **Feature Extraction:**  
  - Backbone: **ResNet-50** (pretrained on ImageNet, fine-tuned for embeddings)  
  - Output: Image feature vector  

---

#### 🔗 Multimodal Fusion and Prediction
- **Fusion:** Concatenation of text and image embeddings  
- **Model:** LightGBM  
- **Training Objective:** Minimize SMAPE  
- **Framework:** LightGBM v4.6 (custom handling for early stopping and evaluation logging)

---

## 4. Model Performance

| Metric | Validation Result |
|:-------:|:----------------:|
| **SMAPE** | **63.79%** |

The hybrid model showed consistent improvement over unimodal baselines:
- Text-only TF–IDF: ~68% SMAPE  
- Image-only CNN embeddings: ~71% SMAPE  
- Text + Image (Hybrid): **63.79% SMAPE**

---

## 5. Conclusion
Our **TF–IDF + ResNet-50 + LightGBM** hybrid system successfully captures complementary textual and visual patterns.  
Key takeaways:
- **Text preprocessing** and **numeric token retention** are critical for modeling product price semantics.  
- **Visual embeddings** enhance differentiation for similar text descriptions.  
- **LightGBM** efficiently models non-linear interactions between modalities.  

Overall, the approach delivers **high accuracy**, **interpretability**, and **computational efficiency** — aligning well with real-world catalog pricing scenarios.

---

## 6. Tech Stack
| Component | Tool/Library |
|:-----------|:-------------|
| Text Processing | Scikit-learn (TF–IDF) |
| Image Processing | PyTorch (ResNet-50) |
| Model Fusion | NumPy, Pandas |
| Regression Model | LightGBM |
| Evaluation Metric | SMAPE |
| Environment | Python 3.10 |

---

## 7. Future Work
- Incorporate transformer-based embeddings (e.g., **BERT**) for richer textual understanding.  
- Introduce visual attention mechanisms to focus on relevant product regions.  
- Explore end-to-end multimodal training with joint optimization.

---
