# Amazon Product Recommendation System

This project implements multiple recommendation engine architectures to provide personalized product suggestions using the Amazon Electronics dataset. As a Data Science project, it explores the transition from simple popularity-based ranking to advanced collaborative filtering and matrix factorization techniques.

## 🚀 Overview
In modern e-commerce, personalized recommendations are essential for engagement. This project simulates the role of a Data Science Manager at Amazon, tasked with building a system to recommend products based on historical user rating data.

## 🛠️ Technical Stack
* **Language:** Python
* **Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`
* **ML Framework:** `scikit-surprise` (for KNN and SVD models)
* **Environment:** Google Colab / Jupyter Notebook

## 📊 Models Implemented
The repository contains three distinct recommendation strategies:

1.  **Rank-Based Recommendation System**
    * Targets the "Cold Start" problem for new users.
    * Recommends products based on highest average ratings and minimum interaction thresholds.

2.  **Collaborative Filtering (User-User & Item-Item)**
    * **User-User:** Finds similar users based on their rating patterns using Cosine Similarity and KNN.
    * **Item-Item:** Recommends products similar to those a user has already liked.
    * Includes hyperparameter tuning via `GridSearchCV` to optimize $k$ neighbors and similarity metrics.

3.  **Model-Based Collaborative Filtering (SVD)**
    * Uses **Singular Value Decomposition (SVD)** to identify latent features in the user-item matrix.
    * Handles sparse data more effectively than basic similarity models.

## 📈 Key Results
The models were evaluated using **RMSE**, **Precision@k**, **Recall@k**, and **F1-Score**:
* **Baseline SVD RMSE:** 0.9217
* **Optimized SVD F1-Score:** 0.878
* **Optimized User-User F1-Score:** 0.884

## 📂 Dataset Details
The dataset used contains over 7.8 million observations. To ensure computational efficiency and recommendation quality, the data was filtered to include:
* **Users** with at least 50 ratings.
* **Products** with at least 5 ratings.

## 💡 How to Use
1. Clone the repository.
2. Ensure `scikit-surprise` is installed: `pip install surprise`.
3. Open `recommendation_systems_learner_notebook_full_code_-1-2.ipynb` in Google Colab or Jupyter.
4. Run the cells to see exploratory data analysis and model performance comparisons.
