from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.optimize import differential_evolution
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pickle
import os

# PATH
ITERATION_CACHE_PATH = "./data/iteration_results.pkl"
CLUSTER_RESULT_PATH = "./data/cluster_result.pkl"

# Variabel global iterasi
iteration_results = []

# Save pickle
def save_to_pickle(obj, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)

# Load pickle
def load_from_pickle(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

# Vectorize TF-IDF
def vectorize_text(text_series):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(text_series)
    return X, vectorizer

# Objective function untuk Differential Evolution
def objective(params, X):
    eps, min_samples = params
    if eps <= 0 or min_samples < 1:
        score = 1
    else:
        db = DBSCAN(eps=eps, min_samples=int(min_samples), metric='cosine')
        labels = db.fit_predict(X.toarray())

        if len(set(labels)) <= 1 or len(set(labels)) == len(X.toarray()):
            score = 1
        else:
            score = -silhouette_score(X, labels, metric='cosine')

    iteration_results.append({
        'eps': eps,
        'min_samples': int(min_samples),
        'silhouette_score': -score if score != 1 else None
    })

    return score

# Optimasi DBSCAN With Differential Evolution
def optimize_dbscan(X, iteration_cache_path = ITERATION_CACHE_PATH):
    global iteration_results
    iteration_results = []
    bounds = [(0.1, 1.5), (2, 10)]

    result = differential_evolution(objective, bounds, args=(X,), strategy='best1bin', maxiter=50, popsize=15)
    save_to_pickle(iteration_results, iteration_cache_path)
    return result.x

# Terapkan DBSCAN
def apply_dbscan(X, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=int(min_samples), metric='cosine')
    labels = dbscan.fit_predict(X.toarray())
    return labels

# Plot PCA Clustering
def plot_pca(X, labels):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X.toarray())
    plt.figure(figsize=(8,6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
    plt.title('DBSCAN Clustering')
    plt.colorbar()
    st.pyplot(plt)
    plt.clf()
