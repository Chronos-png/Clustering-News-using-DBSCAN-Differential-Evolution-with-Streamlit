import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import os

# EDA helper functions
factory_stopword = StopWordRemoverFactory()
stopword = set(factory_stopword.get_stop_words())
factory_stemmer = StemmerFactory()
stemmer = factory_stemmer.create_stemmer()

def clean_text(text):
    text = str(text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def token_stopword(text):
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopword]
    return tokens

def stemming(tokens):
    return [stemmer.stem(word) for word in tokens]

def showHistogram(df, column, title):
    plt.figure(figsize=(10, 5))
    sns.histplot(df[column], bins=30, kde=True)
    plt.title(title)
    plt.xlabel('Jumlah Kata')
    plt.ylabel('Frekuensi')
    return plt.gcf()  # Return the current figure for Streamlit to display

# Caching EDA result biar ga hitung ulang
@st.cache_data
def run_eda(file_path):
    df = pd.read_csv(file_path)
    st.write("📥 Load Dataset Scraping Berita Detik")
    st.dataframe(df.head())

    info = {}

    # Jumlah baris dan kolom
    info['rows'], info['cols'] = df.shape

    # Data Null
    info['missing_values'] = df.isnull().sum()

    # Drop null
    df = df.dropna()
    info['after_dropna'] = df.shape[0]
    info['title_duplicates'] = df.duplicated(subset='judul').sum()
    info['content_duplicates'] = df.duplicated(subset='isi').sum()

    # Save info after drop null
    info['after_missing_values'] = df.isnull().sum()

    # Drop Duplikat
    df = df.drop_duplicates(subset='judul')
    df = df.drop_duplicates(subset='isi')
    info['after_dedup'] = df.shape[0]

    # Panjang teks
    df['panjang_isi'] = df['isi'].apply(lambda x: len(x.split()))
    df['panjang_judul'] = df['judul'].apply(lambda x: len(x.split()))
    histogram_judul = showHistogram(df, 'panjang_judul', 'Histogram Panjang Teks Judul')
    histogram_isi = showHistogram(df, 'panjang_isi', 'Histogram Panjang Teks Isi')

    # Cleaning teks
    df['judul'] = df['judul'].apply(clean_text)
    df['isi'] = df['isi'].apply(clean_text)

    # Tokenisasi, stopword removal, stemming
    df['judul_tokens'] = df['judul'].apply(token_stopword)
    df['isi_tokens'] = df['isi'].apply(token_stopword)
    df['judul_tokens'] = df['judul_tokens'].apply(stemming)
    df['isi_tokens'] = df['isi_tokens'].apply(stemming)

    # Word frequency
    all_text = ' '.join([' '.join(tokens) for tokens in df['isi_tokens']])
    words = all_text.split()
    counter = Counter(words)
    info['top_words'] = counter.most_common(20)

    # Wordcloud
    wordcloud_fig = plt.figure(figsize=(10, 5))
    wc = WordCloud(width=800, height=400, max_font_size=110, background_color='white').generate(all_text)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')

    # TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['isi'])
    info['tfidf_shape'] = X.shape
    sparsity = (X.nnz / (X.shape[0] * X.shape[1])) * 100
    info['sparsity'] = sparsity

    # PCA plot
    X_dense = X.toarray()
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_dense)
    pca_fig = plt.figure(figsize=(6, 5))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], color='blue')
    plt.title("PCA Scatter Plot")

    # Cosine distance
    distances = cosine_distances(X_dense)
    dist_fig = plt.figure(figsize=(6, 5))
    sns.histplot(distances.flatten(), bins=30)
    plt.title("Distribusi Jarak Cosine")

    return info, wordcloud_fig, pca_fig, dist_fig, histogram_judul, histogram_isi

# Streamlit EDA page
def eda_page(file_path = None):
    st.title("Eksplorasi Data Awal")
    if file_path is None:
        file_path = './data/scrap_berita_detik.csv'

    if not os.path.exists(file_path):
        st.error("❌ File tidak ditemukan.")
    else:
        info, wordcloud_fig, pca_fig, dist_fig, histogram_judul, histogram_isi = run_eda(file_path)

        # Data Info
        st.write(f"**Jumlah Data Awal:** {info['rows']} baris, {info['cols']} kolom")

        # Missing Value Show
        st.subheader("Informasi Data Missing Values")
        st.write("**Jumlah Missing Values per Kolom:**")
        st.dataframe(info['missing_values'])

        # After Drop Null Show
        st.subheader("Informasi Data Setelah Drop Null")
        st.write("**Jumlah Missing Values Setelah Drop Missing:**")
        st.dataframe(info['after_missing_values'])
        st.write(f"**Jumlah Data Setelah Drop Null:** {info['after_dropna']} baris")

        # Duplikat Show
        st.subheader("Informasi Data Duplikat")
        st.write(f"**Jumlah Duplikat Judul:** {info['title_duplicates']} baris")
        st.write(f"**Jumlah Duplikat Isi:** {info['content_duplicates']} baris")
        st.write(f"**Jumlah Data Setelah Hapus Duplikasi:** {info['after_dedup']} baris")

        # Panjang Teks Show
        st.subheader("Panjang Teks Judul dan Isi")
        st.pyplot(histogram_judul)  # Histogram Panjang Teks Judul
        st.pyplot(histogram_isi)    # Histogram Panjang Teks Isi

        # Analisis Teks
        st.subheader("Analisis Frekuensi Kata")
        st.write("**Top 20 Kata Paling Sering Muncul:**")
        st.dataframe(info['top_words'])

        st.subheader("WordCloud")
        st.pyplot(wordcloud_fig)

        st.subheader("Distribusi Pola Cluster Alami Data Sebelum Di-cluster (PCA Scatter Plot)")
        st.pyplot(pca_fig)

        st.subheader("Distribusi Jarak Cosine antar Dokumen")
        st.pyplot(dist_fig)
