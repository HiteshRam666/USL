# Prac 1: Data Augmentation: What, Why, and How  

## 📌 What is Data Augmentation?  
Data Augmentation is a technique used in machine learning and deep learning to artificially increase the size of a dataset by creating modified versions of existing data. This is especially useful in image processing, where transformations such as rotation, flipping, zooming, and brightness adjustments can be applied to generate more training samples.  

## 🎯 Why Use Data Augmentation?  
✔ **Prevents Overfitting** – Helps the model generalize better by exposing it to more variations of the same data.  
✔ **Improves Model Performance** – Increases the diversity of training samples, leading to better accuracy.  
✔ **Compensates for Small Datasets** – Augmentation can help when collecting more real-world data is difficult or expensive.  
✔ **Enhances Robustness** – The model learns to handle different perspectives, lighting conditions, and distortions.  

## 📢 Conclusion  
Data augmentation is a powerful way to improve model accuracy and robustness, particularly in computer vision tasks. By applying transformations such as rotation, zoom, and flipping, models can better generalize to unseen data.  

---

# **Prac 2: Semi-Supervised Learning with Label Propagation📌**  

## **🛠 What is Semi-Supervised Learning?**  
Semi-supervised learning is a machine learning approach that **combines labeled and unlabeled data** to improve model performance. It is useful when:  
- **Labeled data is limited** or expensive to obtain.  
- **Unlabeled data is abundant**, and we want to use it to enhance learning.  
- **The model can make confident predictions** on some unlabeled data, which can then be used for further training.  

In this repository, we use **XGBoost** and **Label Propagation** to iteratively label high-confidence samples from an initially unlabeled dataset.  

---

## **🔍 What is Label Propagation?**  
Label Propagation is a **self-training technique** where:  
1. A model is trained on the available **labeled dataset**.  
2. The model predicts labels for **unlabeled samples**.  
3. Only **high-confidence predictions** are added to the labeled dataset.  
4. The process **repeats iteratively** until no more high-confidence samples remain.  

This approach **gradually expands the labeled dataset**, leading to **better model performance** over time.

---

## **🚀 Why Use Semi-Supervised Learning?**  
✔️ **Reduces labeling effort** – You don’t need to label all data manually.  
✔️ **Utilizes available data** – Makes use of the large pool of unlabeled data.  
✔️ **Improves model performance** – The model learns from additional confidently predicted samples.  
✔️ **Real-world applications** – Used in NLP, medical imaging, fraud detection, and more.

---

## Prac 3: **Dimensionality Reduction & Feature Selection in Machine Learning**  

## **📌 Introduction**  
Dimensionality reduction is a crucial technique in machine learning used to reduce the number of input variables (features) in a dataset while preserving important information. It helps in improving model performance, reducing computation time, and avoiding overfitting.

In this project, we perform **dimensionality reduction** using **feature selection** and **feature extraction** techniques and train models to evaluate their effectiveness.

---

## **🧩 The Curse of Dimensionality**  
The **Curse of Dimensionality** refers to the challenges that arise when working with high-dimensional data. Some of the major issues include:  

- **Increased computation time:** Training models on high-dimensional data is computationally expensive.  
- **Overfitting:** Models may learn noise instead of patterns due to too many features.  
- **Data sparsity:** In high dimensions, data points become sparse, making distance-based models ineffective.  

### **🛠️ Solution: Dimensionality Reduction**  
To combat the curse of dimensionality, we can apply **dimensionality reduction techniques** like:  

✅ **Feature Selection:** Selecting the most important features that contribute to the target variable.  
✅ **Feature Extraction:** Transforming data into a lower-dimensional space while preserving meaningful information.  

---

## **🔬 Feature Selection Methods**
Feature selection helps in reducing irrelevant or redundant features while maintaining the predictive power of the dataset.  
This project uses the following techniques:  

1️⃣ **Variance Threshold:** Removes features with low variance (almost constant features).  
2️⃣ **SelectKBest (ANOVA F-score):** Selects top features based on their statistical importance.  

### **📌 Feature Selection Solutions**
✔ **Remove redundant features:** Use correlation heatmaps to detect highly correlated features.  
✔ **Use domain knowledge:** Identify the most relevant features.  
✔ **Apply automated feature selection methods:** Such as SelectKBest, Recursive Feature Elimination (RFE), and LASSO regression.  

---

## **🎭 Feature Extraction (Dimensionality Reduction Using PCA)**
Feature extraction transforms high-dimensional data into a lower-dimensional space. This project implements:  

📌 **Principal Component Analysis (PCA)**:  
- Reduces dimensionality by projecting data into a new feature space while retaining maximum variance.  
- Helps in visualizing and improving model efficiency.  

---
