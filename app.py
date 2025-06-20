import streamlit as st
import pandas as pd
from preprocessing import preprocess_data
from clustering import (
    vectorize_text, optimize_dbscan, apply_dbscan, 
    plot_pca, save_to_pickle, load_from_pickle,
    ITERATION_CACHE_PATH, CLUSTER_RESULT_PATH
)
import os
from EDA import eda_page
from sklearn.metrics import silhouette_score

# Path file
FILE_PATH = './data/scrap_berita_detik.csv'
PREPROCESS_PATH = './output/preprocessed_dataset_detik.csv'

# Sidebar Menu
st.sidebar.title("Menu Navigasi")
menu = st.sidebar.radio("Pilih Halaman:", ["Home", "EDA Data", "Preprocessing", "New Clustering"])

# Caching data loader
@st.cache_data
def load_csv(file_path):
    return pd.read_csv(file_path)

# Caching preprocessing
@st.cache_data
def load_and_preprocess(data):
    return preprocess_data(data)

# Clustering function
def cluster_data(df):
    if df.shape[0] < 2:
        st.error("❌ Data tidak cukup untuk clustering.")
        return None, None, None, None

    if os.path.exists(CLUSTER_RESULT_PATH) and os.path.exists(ITERATION_CACHE_PATH):
        st.success("📦 Memuat hasil clustering dari cache.")
        X, vectorizer = vectorize_text(df['isi_final'])
        eps_opt, min_samples_opt, labels = load_from_pickle(CLUSTER_RESULT_PATH)
        results_df = pd.DataFrame(load_from_pickle(ITERATION_CACHE_PATH))
    else:
        st.write("🔍 Melakukan vektorisasi TF-IDF...")
        X, vectorizer = vectorize_text(df['isi_final'])
        st.write("✅ Vektorisasi selesai.")

        st.write("⚙️ Optimasi parameter DBSCAN (DE)...")
        eps_opt, min_samples_opt = optimize_dbscan(X)
        labels = apply_dbscan(X, eps_opt, min_samples_opt)
        save_to_pickle((eps_opt, min_samples_opt, labels), CLUSTER_RESULT_PATH)
        results_df = pd.DataFrame(load_from_pickle(ITERATION_CACHE_PATH))

    st.subheader("📋 Hasil Iterasi Optimasi")
    results_df = results_df.dropna()
    st.dataframe(results_df.reset_index(drop=True), use_container_width=True)

    st.write("📊 Hasil Akhir Optimasi:")
    st.write(f"Optimal eps: {eps_opt:.4f}, min_samples: {min_samples_opt:.0f}")
    
    if len(set(labels)) > 1:
        sil_score = silhouette_score(X, labels, metric='cosine')
    else:
        sil_score = None  # kalau cluster cuma 1 / semuanya noise
     
    
    if sil_score is not None:
        st.write(f"Silhouette Score Akhir: {sil_score:.4f}")
    else:
        st.write("Silhouette Score tidak dapat dihitung (cluster tunggal atau noise).")   
    
    return X, labels, eps_opt, min_samples_opt

# Home
if menu == "Home":
    st.title("Klasterisasi Berita Online (DBSCAN + Differential Evolution)")

    if not os.path.exists(FILE_PATH):
        st.error("❌ File scrap_berita_detik.csv tidak ditemukan.")
    else:
        raw_data = load_csv(FILE_PATH)
        st.write("📥 Data Awal")
        st.write(raw_data.head())

        if os.path.exists(PREPROCESS_PATH):
            df = load_csv(PREPROCESS_PATH)
        else:
            df = load_and_preprocess(raw_data)

        st.write("✅ Data setelah preprocessing:")
        st.write(df.head())

        X, labels, eps_opt, min_samples_opt = cluster_data(df)
        if labels is not None:
            df['cluster_opt'] = labels

            st.subheader("📊 Hasil Klaster:")
            st.write(df[['isi', 'cluster_opt']])

            st.write("📈 Visualisasi PCA Clustering:")
            plot_pca(X, labels)

# EDA
elif menu == "EDA Data":
    eda_page()

# Preprocessing manual
elif menu == "Preprocessing":
    st.title("Preprocessing Data")
    data = pd.read_csv(FILE_PATH)
    df = preprocess_data(data)
    st.write("Data Setelah Preprocessing:", df.head())
    
# New Clustering
elif menu == "New Clustering":
    st.title("Buat Cluster Baru Menggunakan DBSCAN + Differential Evolution")
    uploaded_file = st.file_uploader("Upload file CSV berita:", type="csv")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("📥 Data Awal:")
        st.dataframe(data.head())

        if st.button("Mulai EDA"):
            temp_path = f'./data/temp_uploaded.csv'
            data.to_csv(temp_path, index=False)
            eda_page(temp_path)

        if st.button("Mulai Clustering"):
            st.subheader("🔄 Proses Preprocessing Data")
            df = preprocess_data(data)
            st.write("✅ Data Setelah Preprocessing:")
            st.dataframe(df.head())

            if df.shape[0] < 2:
                st.error("❌ Data tidak cukup untuk clustering.")
            else:
                st.write("🔍 Melakukan vektorisasi TF-IDF...")
                X, vectorizer = vectorize_text(df['isi_final'])
                st.success("✅ Vektorisasi selesai.")

                iteration_cache_new = './output/iteration_results_new.pkl'

                st.write("⚙️ Optimasi parameter DBSCAN (DE)...")
                eps_opt, min_samples_opt = optimize_dbscan(X, iteration_cache_path=iteration_cache_new)
                st.success("✅ Optimasi selesai.")

                # Load hasil iterasi dari pickle
                results_df = pd.DataFrame(load_from_pickle(iteration_cache_new))
                results_df = results_df.dropna().sort_values(by='silhouette_score', ascending=False)

                st.subheader("📋 Hasil Iterasi Optimasi")
                st.dataframe(results_df.reset_index(drop=True), use_container_width=True)

                st.write("📊 Hasil Akhir Optimasi:")
                st.write(f"Optimal eps: {eps_opt:.4f}, min_samples: {min_samples_opt:.0f}")

                labels = apply_dbscan(X, eps_opt, min_samples_opt)
                df['cluster_opt'] = labels

                if len(set(labels)) > 1:
                    sil_score = silhouette_score(X, labels, metric='cosine')
                else:
                    sil_score = None  # kalau cluster cuma 1 / semuanya noise

                st.subheader("📊 Hasil Klaster:")
                st.dataframe(df[['isi', 'cluster_opt']])

                st.write("📈 Visualisasi PCA Clustering:")
                plot_pca(X, labels)

