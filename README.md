# Principal Component Analysis (PCA) Formative Assignment

## Overview

This project implements **Principal Component Analysis (PCA) from scratch** on real-world health indicator data. It demonstrates key machine learning concepts including data preprocessing, dimensionality reduction, and performance optimization.

**Student:** Nshuti Shalom Silver Jr  
**Dataset:** World Health Organization (WHO) Health Indicators for Cameroon  
**Completed:** February 2025

---

## 📊 Dataset

**Dataset Name:** `health_indicators_cmr.csv`  
**Source:** WHO Global Health Observatory (GHO)  
**Records:** 22,952 entries (after cleaning)  
**Features:** 625 features (after encoding)

### Key Characteristics:
- **Domain:** Public Health & Epidemiology
- **Geographic Focus:** Cameroon, Africa
- **Health Indicators Include:**
  - Mortality rates (under-five, neonatal, maternal)
  - Disease prevalence (malaria, tuberculosis, diabetes)
  - Nutritional indicators (stunting, wasting)
  - Environmental health metrics
  - Education & socioeconomic dimensions

### Data Challenges Addressed:
✅ Missing values (2,478 missing numeric values)  
✅ Non-numeric columns (indicator names, URLs, categorical variables)  
✅ Mixed data types requiring preprocessing  
✅ Large feature dimensionality after encoding

---

## 🎯 Project Goals

1. **Implement PCA from scratch** without using scikit-learn's PCA
2. **Handle real-world data quality issues:**
   - Remove metadata rows
   - Convert data types appropriately
   - Handle missing values
   - Encode categorical variables
3. **Select optimal number of components** based on explained variance
4. **Visualize dimensionality reduction** effects
5. **Benchmark performance** against alternative methods

---

## 🔧 Technical Implementation

### Step 1: Data Loading & Exploration
- Load 22,953 rows and 17 columns from CSV
- Identify metadata rows and remove them
- Convert numeric columns to float64 type

### Step 2: Data Preprocessing
- Drop rows with missing numeric values (target variable)
- One-hot encode categorical features:
  - GHO (CODE) - indicator type
  - DIMENSION (TYPE) - breakdown category (e.g., SEX, AGEGROUP)
  - DIMENSION (CODE) - specific breakdown
  - DIMENSION (NAME) - readable labels
- Keep YEAR (DISPLAY) as numeric feature
- **Result:** 20,474 rows × 625 features

### Step 3: Standardization
- Zero-mean normalization: `X_normalized = X - mean`
- Unit variance scaling: `X_scaled = X_normalized / std`
- **Verification:** Column means ≈ 0, column stds ≈ 1

### Step 4: Eigendecomposition
- Compute covariance matrix: 625 × 625
- Perform eigenvalue decomposition using `np.linalg.eig()`
- Sort eigenvalues in descending order
- Extract corresponding eigenvectors

### Step 5: Component Selection
Analyzed cumulative explained variance at different thresholds:

| Threshold | Components Needed | Actual Variance |
|-----------|-------------------|-----------------|
| 70% | 268 | 70.14% |
| 80% | 330 | 80.07% |
| **90%** | **393** | **90.15%** |
| 95% | 424 | 95.11% |

**Selected:** 393 components for 90% variance retention

### Step 6: Dimensionality Reduction
- Project standardized data onto top 393 principal components
- Result: 20,474 × 393 feature matrix
- Dimensionality reduced by **37%** while retaining 90% of variance

### Step 7: Visualization & Analysis
- **Before PCA:** Scattered distribution of scaled year vs. numeric values
- **After PCA:** Points projected onto PC1 (max variance) and PC2 (second-max variance)
- **Effect:** Data reoriented while preserving cluster structure

### Step 8: Performance Optimization
Benchmarked computation methods:
- **Eigendecomposition (np.linalg.eig):** 0.8940 seconds
- **SVD (np.linalg.svd):** 2.7356 seconds
- **Recommendation:** Eigendecomposition more efficient for this dataset size

---

## 📈 Results Summary

### Dimensionality Reduction Success
- **Original Features:** 625
- **Reduced Features:** 393
- **Reduction:** 37% fewer dimensions
- **Information Retained:** 90.15% of variance

## Installation & Usage

### Requirements
```bash
pip install -r requirements.txt
