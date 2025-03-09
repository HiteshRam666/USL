# Prac 1: Data Augmentation:

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

# Prac 3: **Dimensionality Reduction & Feature Selection in Machine Learning**  

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

# Prac 4: Principal Component Analysis (PCA) 

## 📌 What is PCA?  
Principal Component Analysis (PCA) is a dimensionality reduction technique used in machine learning and statistics. It transforms high-dimensional data into a lower-dimensional form while preserving as much variance as possible.  

### 🔹 Key Concepts:  
- **Dimensionality Reduction**: Helps reduce the number of features while retaining essential patterns.  
- **Feature Extraction**: Converts correlated features into uncorrelated principal components.  
- **Variance Maximization**: The principal components capture the highest variance in the data.  

---

## ❓ Why PCA?  
PCA is widely used for:  
✔ **Reducing Computational Cost**: Lower dimensions mean faster processing.  
✔ **Handling Multicollinearity**: Removes redundant features.  
✔ **Visualization**: Helps in plotting high-dimensional data in 2D or 3D.  
✔ **Noise Reduction**: Filters out irrelevant variations in data.  

---

## 📊 PCA Calculation Steps  
### Step 1: Standardizing the Data  
Subtract the mean from each feature to center the data around zero.  

### Step 2: Compute the Covariance Matrix  
Calculate the covariance between features to understand their relationships.  

![image](https://github.com/user-attachments/assets/bdaea204-0c17-459a-926b-debd50a24034)

### Step 3: Compute Eigenvalues and Eigenvectors  
Eigenvalues indicate variance captured by each principal component, while eigenvectors define new feature axes.  

![image](https://github.com/user-attachments/assets/c0a3ec2f-ffba-42a7-9720-f7422db52e1c)

### Step 4: Sort Eigenvalues in Descending Order  
Select the top **k** eigenvectors corresponding to the **k** largest eigenvalues.  

### Step 5: Transform Data  
Multiply the original data matrix by the selected eigenvectors to obtain the principal components.  

### 📌 **Covariance Matrix, Eigenvalues, and Eigenvectors – Short Explanation**  

#### **1️⃣ Covariance Matrix**  
A **covariance matrix** is a square matrix that represents the relationships (covariances) between multiple features in a dataset. It helps measure how two features vary together.  

- If **covariance is positive**, both features increase together.  
- If **covariance is negative**, one increases while the other decreases.  
- If **covariance is zero**, the features are independent.  

#### **2️⃣ Eigenvalues and Eigenvectors**  
Eigenvalues and eigenvectors help in transforming data into new principal components.  

✅ **Eigenvector**: A direction in the feature space along which the data varies the most.  
✅ **Eigenvalue**: The magnitude (amount of variance) along the corresponding eigenvector.  

---

🚀 **In short**:  
- **Covariance Matrix**: Measures feature relationships.  
- **Eigenvalues**: Show how much variance a principal component captures.  
- **Eigenvectors**: Define the new directions (principal components) in transformed space.  
---

# Prac 5: **📌 K-Means Clustering:**

### **1️⃣ What is K-Means?**
K-Means is an **unsupervised machine learning algorithm** used for **clustering data** into **K groups** based on feature similarity. It aims to minimize the distance between data points and their assigned cluster centers (centroids).  

### **2️⃣ How K-Means Works? (Workflow)**
The algorithm follows these steps:

1️⃣ **Choose the number of clusters (K)**.  
2️⃣ **Randomly initialize K centroids** (initial cluster centers).  
3️⃣ **Assign each data point to the nearest centroid** (based on Euclidean distance).  
4️⃣ **Recompute centroids** by taking the mean of all points in each cluster.  
5️⃣ **Repeat steps 3-4 until centroids remain stable** (i.e., no major changes in cluster assignments).  

**📌 Final Output:** Data points are grouped into K clusters.  

---

## **📊 Choosing the Right Number of Clusters (K)**
Choosing the optimal number of clusters is crucial in K-Means. Two key methods for this are:  

### **1️⃣ Elbow Method**
The **Elbow Method** helps determine the best K by plotting the **Within-Cluster Sum of Squares (WCSS)** for different K values and identifying the "elbow point" where adding more clusters **does not significantly decrease WCSS**.

#### **🔹 Within-Cluster Sum of Squares (WCSS)**
WCSS measures how compact the clusters are by calculating the sum of squared distances between each data point and its centroid.  
📌 **Formula:**  

![image](https://github.com/user-attachments/assets/99c377ee-66d8-43b6-abfc-19f7c283b946)


- **Lower WCSS = Better clustering**  

#### **📌 Elbow Method Steps**
1. Compute WCSS for different values of K (e.g., 1 to 10).
2. Plot K vs. WCSS.
3. Find the "elbow point" (where WCSS stops decreasing significantly).
4. Choose K at this elbow.

---

### **2️⃣ Silhouette Score**
The **Silhouette Score** measures **how well-separated** the clusters are. It evaluates each point by comparing:
- **Intra-cluster distance (a)** → Distance to points in the same cluster.  
- **Inter-cluster distance (b)** → Distance to the nearest cluster.  

📌 **Formula:**  
![image](https://github.com/user-attachments/assets/40d3ff50-3615-46b8-a934-8ad54b94ccc5)


Silhouette Score helps validate the chosen **K** by checking cluster separation.

---

# Prac 6: Hierarchical Clustering: A Comprehensive Guide  

#### 📌 What is Hierarchical Clustering?  
Hierarchical clustering is an unsupervised machine learning algorithm used to group data points into clusters based on their similarity. Unlike K-Means, it does not require the number of clusters to be predefined and creates a hierarchy of clusters in a tree-like structure called a **dendrogram**.  

---

#### 🔍 Why Use Hierarchical Clustering?  
1. **No need to predefine clusters** 🧐 – Unlike K-Means, it doesn't require specifying `k` beforehand.  
2. **Dendrogram analysis** 📊 – Provides a visual representation of how clusters are formed.  
3. **Interpretable clustering** ✅ – Good for understanding relationships in data.  
4. **Works well with small datasets** 📉 – Computationally expensive for large datasets but effective for smaller ones.  

---

#### ⚙️ Workflow of Hierarchical Clustering  
1. **Compute pairwise distances** between data points.  
2. **Create a linkage matrix** to measure distances between clusters.  
3. **Build a dendrogram** that visually represents merging clusters.  
4. **Determine the optimal number of clusters** by cutting the dendrogram at an appropriate level.  
5. **Assign cluster labels** to the dataset.  

---

#### 🔗 Types of Linkage in Hierarchical Clustering  
Linkage defines how distances between clusters are calculated.  

1. **Single Linkage (Minimum Linkage)** – Distance between the closest points in two clusters.  
2. **Complete Linkage (Maximum Linkage)** – Distance between the farthest points in two clusters.  
3. **Average Linkage** – Average of all pairwise distances between points in two clusters.  
4. **Centroid Linkage** – Distance between centroids of clusters.  
5. **Ward's Linkage** – Minimizes variance within clusters (default in `scikit-learn`).  

---

#### 📊 Dendrogram and Cluster Selection  
- The **dendrogram** is a tree-like structure that shows how clusters are merged.  
- To determine the optimal number of clusters, look for the **longest vertical line** that can be cut without crossing horizontal merges.  

---

#### 📈 Next Steps After Clustering  
- **Evaluate cluster quality** using metrics like **Silhouette Score**.  
- **Compare with K-Means clustering** for performance insights.  
- **Visualize in 3D** if more than two features are present.  
- **Apply feature scaling (StandardScaler/MinMaxScaler)** to improve accuracy.  
- **Use domain knowledge** to interpret cluster characteristics.  

---

### 🚀 Conclusion  
Hierarchical clustering is a powerful technique for grouping data based on similarity without predefining clusters. Its dendrogram representation helps in understanding data structure, but its high computational cost makes it less suitable for large datasets.  

---

# Prac 7: **🌀 DBSCAN (Density-Based Spatial Clustering)**
DBSCAN is an **unsupervised** clustering algorithm that groups nearby points while identifying **outliers** as noise. Unlike **K-Means** or **Hierarchical Clustering**, DBSCAN **does not require** specifying the number of clusters beforehand and is **robust to outliers**. 🚀  

---

## **🔍 How DBSCAN Works**
DBSCAN groups points based on two key parameters:  
1. **🔹 eps (ε)** – The maximum distance between two points to be considered neighbors.  
2. **🔹 min_samples** – The minimum number of points required in a neighborhood to form a **dense region (cluster).**  

### **📌 Steps of DBSCAN**
1️⃣ **Select a random unvisited point** and check its neighborhood using **eps**.  
2️⃣ If the number of neighbors **≥ min_samples**, a **new cluster** is formed.  
3️⃣ Expand the cluster by recursively adding **density-reachable points**.  
4️⃣ If a point has fewer than **min_samples** neighbors, it is labeled as **noise**. ❌  
5️⃣ Repeat until all points are visited. ✅  

---

## **💡 Benefits of DBSCAN**
✅ **No need to specify the number of clusters** – Unlike K-Means, which requires a predefined `k`.  
✅ **Handles outliers well** – Unlike K-Means and Hierarchical Clustering, DBSCAN can mark noise points. 🏝️  
✅ **Works with arbitrary-shaped clusters** – Can identify **non-spherical** clusters, unlike K-Means which assumes spherical clusters. 🔄  
✅ **Robust to varying densities** – Can handle clusters of **different densities**. 📊  

---

## **⚖️ Comparison with Other Clustering Algorithms**
| Feature          | 🔹 K-Means               | 🔹 Hierarchical Clustering | 🔹 DBSCAN |
|-----------------|----------------------|-------------------------|--------|
| **Clusters shape** | 🔵 Spherical            | 🌲 Tree-based               | 🔄 Arbitrary |
| **Handles outliers?** | ❌ No               | ❌ No                     | ✅ Yes |
| **Number of clusters?** | Must be specified | Must be specified        | 📌 Found automatically |
| **Scalability** | ⚡ Fast | 🐢 Slow | ⚡ Fast (for medium datasets) |

---

## **🛠️ When to Use DBSCAN?**
📌 When you **don’t know the number of clusters** in advance.  
📌 When data has **arbitrary-shaped clusters** (not just circular).  
📌 When handling **outliers/noise** is important. 🚧  
📌 When dealing with **density-based clustering problems**.  

---

## **🚀 Conclusion**
DBSCAN is a **powerful** clustering algorithm that excels in detecting clusters of **arbitrary shapes** while identifying **noise**. It is particularly useful when the number of clusters is **unknown** and when dealing with datasets containing **outliers**.  
