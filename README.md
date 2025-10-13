# ML Challenge 2025: Smart Product Pricing Solution

**Team Name:** Synaptech 

**Team Members:** Viraj Kumar, Amon Harsh, Ansh Arya, Sarvagya Dwivedi

**Submission Date:** 13/10/25

---

## 1. Executive Summary
Our solution predicts e-commerce product prices using a **multimodal fusion approach** that combines **text and image embeddings**. We integrate **transformer-based NLP (all-mpnet-base-v2)** and **vision models (CLIP)** with a **custom neural regression network**, optimized using a differentiable SMAPE loss. This enables robust and semantically aligned price prediction.

---

## 2. Methodology Overview

### 2.1 Problem Analysis
The challenge requires accurate price prediction for e-commerce products based on **unstructured product descriptions (text)** and **image URLs**.  
Our exploratory data analysis (EDA) revealed strong semantic relationships between textual attributes (e.g., brand, material, product type) and prices, while image features often correlated with product category and quality.
The dataset contains product information in the following fields: 
 - sample_id
 - catalog_text (unstructured product description)
 - image_url (link to product image)
The goal is to predict product prices accurately using Symmetric Mean Absolute Percentage Error (SMAPE) as the evaluation metric.

**Key Observations:**
- Text data contained noise (HTML tags, punctuation, inconsistent casing).
- Image data contained diverse lighting and composition conditions.
- Product descriptions varied in detail; some categories were underrepresented.
- SMAPE metric favors balanced predictions — penalizing both over-and under-predictions.

---

### 2.2 Solution Strategy
Our high-level approach utilizes **multimodal deep learning**, merging **text and image embeddings** to leverage both semantic and visual cues.

**Approach Type:** Hybrid (Multimodal Fusion Model)  
**Core Innovation:** A lightweight, deployable **multimodal fusion architecture** combining **all-mpnet-base-v2 text embeddings** and **CLIP image embeddings**, trained with a **custom differentiable SMAPE loss** to align with leaderboard evaluation.

---

## 3. Model Architecture

### 3.1 Architecture Overview
The model integrates two embedding generators — sentence-transformers/all-mpnet-base-v2 for text and CLIP for images. The resulting embeddings are concatenated and fed into a fully connected neural network for regression.

**Workflow Diagram (Conceptual):**

  ```mermaid
flowchart TD
    A[Catalog Text - Product Description] --> B[Text Preprocessing: regex, lowercase, punctuation]
    B --> C[SentenceTransformer MPNet - Text Embeddings 768d]
    D[Image URL - Product Image] --> E[Image Preprocessing: resize, normalize]
    E --> F[CLIP ViT-B/32 - Image Embeddings 512d]
    C --> G[Concatenate Text and Image Embeddings - 1280d]
    F --> G
    G --> H[Fully Connected Neural Network - BN Dropout ReLU Layers]
    H --> I[Predicted Product Price]
---

### 3.2 Model Components

#### **Text Processing Pipeline**
- **Preprocessing Steps:**
  - Regex-based cleaning
  - Lowercasing and punctuation normalization
  - Removal of extra spaces, special characters, and HTML tags  
- **Model Type:** Sentence Transformer – *all-mpnet-base-v2*  
- **Key Parameters:**
  - Embedding dimension: 3072 
  - Batch size: Tuned for GPU memory (Colab T4)  

#### **Image Processing Pipeline**
- **Preprocessing Steps:**
  - Image resizing and normalization  
  - Center cropping to 224×224  
  - Conversion to RGB and tensor normalization  
- **Model Type:** *OpenAI CLIP (ViT-B/32)*  
- **Key Parameters:**
  - Embedding dimension: 512  
  - Batch normalization and augmentation disabled (for consistency)  

#### **Fusion and Regression Network**
- **Architecture:**
  - Input: Concatenated embeddings (3072 , 512)
  - Layers:
    - Text:
        - Dense(3072) → Dense(1024) → BatchNorm → GeLU → Dropout(0.3)
        - Dense(1024) → Dense(512) → BatchNorm → GeLU
    - Image :
        - Dense(512) → Dense(256) → BatchNorm → GeLU → Dropout(0.3)
        - Dense(256) → Dense(256) → BatchNorm → GeLU 
    - Fusion :
       - Dense(3584) → Dense(512) → BatchNorm → GeLU → Dropout(0.3)
       - Dense(512) → Dense(256)  → GeLU → Dense(1)
- **Loss Function:** Custom differentiable SMAPE  
- **Optimizer:** AdamW   

---

## 4. Model Performance

### 4.1 Validation Results
| Metric | Value |
|:-------|:------|
| **SMAPE Score** | 47.60 |
| **MSE LOSS** | 0.1313 |


**Validation Strategy:**  
- 5-Fold Cross-Validation predictions  
- SMAPE-based monitoring and checkpointing for best fold weights  

---

## 5. Conclusion
Our model achieves competitive SMAPE performance through **transformer-based embeddings** and a **custom neural fusion network**. The design emphasizes **interpretability**, **robustness**, and **efficiency**, making it practical for real-world e-commerce deployment. Key learnings include the importance of semantic text preprocessing and balanced embedding scaling in multimodal fusion.

---

## Appendix

### A. Code Artefacts
**Drive Link:** https://drive.google.com/drive/folders/1Dz4lepBcRw9dl0A8UJXgMtPwah6xhmev?usp=drive_link  
*(Includes data preprocessing scripts, embedding generation notebooks, PyTorch training modules, and inference pipeline.)*

### B. Additional Results
**Drive Link:** https://drive.google.com/drive/folders/1Dz4lepBcRw9dl0A8UJXgMtPwah6xhmev?usp=drive_link
*(Includes Fold-wise SMAPE trend charts and  Training vs. validation loss curves)*


---

**License:**  
Released under the **MIT License** — free for use, modification, and distribution with proper attribution.

---


