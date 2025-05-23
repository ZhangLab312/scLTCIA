# scLTCIA: Single-Cell Long-Tail Class-Incremental Annotation Framework

![scLTCIA Framework](https://github.com/ZhangLab312/scLTCIA/blob/main/images/framwork.png?raw=true)

*Figure 1: The proposed scLTCIA framework for incremental annotation of single-cell data.*

---

## 📜 Abstract

Single-cell type annotation faces critical challenges in real-world scenarios, including catastrophic forgetting, high-dimensional sparsity, and extreme class imbalance. This paper introduces **scLTCIA**, a novel framework that synergizes **distribution-aware generative replay** and **expression-aware knowledge distillation** to enable robust continual learning of novel cell types while preserving knowledge of previously learned types. Through extensive experiments on five scRNA-seq datasets, scLTCIA achieves state-of-the-art performance in both ordered and shuffled incremental annotation tasks, outperforming existing methods by up to **9.47%** in rare-type accuracy.

---

## 🎯 Key Contributions

1. **Distribution-Aware Conditional Diffusion**  
   A novel generative paradigm that reconstructs high-fidelity scRNA-seq data with dynamic sparse masks, preserving biological consistency during incremental replay.
2. **Expression-Aware Multi-View Alignment**  
   A knowledge distillation framework that aligns gene-specific features across scales through attention mechanisms, addressing the high-dimensional sparsity of single-cell data.
3. **Fuzzy Incremental Guidance**  
   A constraint mechanism that suppresses spurious recall of novel cell types in long-tailed distributions, improving annotation robustness.

---

## 🔍 Core Methodology

### Problem Formulation

- **Class-Incremental Annotation**: Sequential introduction of disjoint cell type sets `𝒴_τ`
- **Challenges**:  
  - Long-tailed distribution (e.g., CD34+ cells: 0.29% in Zheng68k)  
  - High-dimensional sparsity (16,384 genes × extreme zero-inflation)  
  - Catastrophic forgetting (base-type accuracy drops ≤3.2% in final sessions)

### Framework Components

1. **Retrospective Generation Module**  
   - Dynamic sparse mask `𝓜_t` guided by expression quantiles  
   - Conditional diffusion with text-encoded cell type embeddings  
2. **Incremental Alignment Module**  
   - Multi-scale attention alignment (`L_attn`)  
   - Semantic consistency constraints (`L_repr`, `L_logit`)  
   - Fuzzy guidance loss for novel-type suppression
