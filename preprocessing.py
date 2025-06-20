import pandas as pd
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import os
import streamlit as st
import io

# Inisialisasi stopword dan stemmer
factory_stopword = StopWordRemoverFactory()
stopword = set(factory_stopword.get_stop_words())

factory_stemmer = StemmerFactory()
stemmer = factory_stemmer.create_stemmer()

# Fungsi load data
def load_data(path):
    return pd.read_csv(path)

# Pembersihan teks
def clean_text(text):
    text = str(text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Tokenisasi + stopword removal
def token_stopword(text):
    tokens = re.findall(r'\b\w+\b', text.lower())
    tokens = [word for word in tokens if word not in stopword]
    return tokens

# Stemming
def stemming(tokens):
    return [stemmer.stem(word) for word in tokens]

# Preprocessing data dengan output ke Streamlit
def preprocess_data(df):
    st.markdown("## 📌 Tahapan Awal Dataset")
    st.subheader("🔍 Data Mentah")
    st.dataframe(df.head())
    st.write(f"Jumlah baris dan kolom: {df.shape}")
    st.markdown("---")

    # Tahap 1: Missing Value
    st.markdown("## 📌 Tahapan Missing Value")
    st.subheader("🔍 Cek Missing Value")
    st.write(df.isnull().sum())

    df = df.dropna()
    st.subheader("🔍 Setelah Drop Missing Value")
    st.dataframe(df.head())
    st.write("Jumlah missing value setelah dibuang:")
    st.write(df.isnull().sum())
    st.write(f"Jumlah baris dan kolom: {df.shape}")
    st.markdown("---")

    # Tahap 2: Duplikat
    st.markdown("## 📌 Tahapan Duplikat")
    st.subheader("🔍 Sebelum Drop Duplikat")
    st.write(f"Jumlah data sebelum drop duplikat: {df.shape}")
    st.dataframe(df.head())

    df = df.drop_duplicates(subset='isi')
    st.subheader("🔍 Setelah Drop Duplikat (kolom 'isi')")
    st.write(f"Jumlah data setelah drop duplikat: {df.shape}")
    st.dataframe(df.head())
    st.markdown("---")

    # Tahap 3: Filter panjang isi
    st.markdown("## 📌 Tahapan Filter Panjang Isi (≥ 50 kata)")
    st.subheader("🔍 Sebelum Filter")
    df['panjang_isi'] = df['isi'].apply(lambda x: len(x.split()))
    st.write(df['panjang_isi'].describe())

    df = df[df['panjang_isi'] >= 50]
    del df['panjang_isi']

    st.subheader("🔍 Setelah Filter Panjang Isi ≥ 50 Kata")
    st.write(f"Jumlah data: {df.shape}")
    st.dataframe(df.head())
    st.markdown("---")

    # Tahap 4: Clean Text
    st.markdown("## 📌 Tahapan Pembersihan Teks")
    st.subheader("🔍 Sebelum Clean Text")
    st.dataframe(df[['isi']].head())

    df['isi'] = df['isi'].apply(clean_text)

    st.subheader("🔍 Setelah Clean Text")
    st.dataframe(df[['isi']].head())
    st.markdown("---")

    # Tahap 5: Tokenisasi + Stopword Removal
    st.markdown("## 📌 Tahapan Tokenisasi + Stopword Removal")
    df['isi_token'] = df['isi'].apply(token_stopword)

    st.subheader("🔍 Hasil Tokenisasi + Stopword Removal")
    st.dataframe(df[['isi_token']].head())
    st.markdown("---")

    # Tahap 6: Stemming
    st.markdown("## 📌 Tahapan Stemming")
    df['isi_stem'] = df['isi_token'].apply(stemming)

    st.subheader("🔍 Hasil Stemming")
    st.dataframe(df[['isi_stem']].head())
    st.markdown("---")

    # Tahap 7: Gabungkan Final Teks
    st.markdown("## 📌 Tahapan Finalisasi Teks")
    df['isi_final'] = df['isi_stem'].apply(lambda x: ' '.join(x))

    st.subheader("🔍 Hasil Gabungan Kata Setelah Stemming")
    st.dataframe(df[['isi_final']].head())
    st.markdown("---")

    # Simpan CSV
    if not os.path.exists('./output'):
        os.makedirs('./output')
    df.to_csv('./output/preprocessed_dataset_detik.csv', index=False)

    # Simpan info struktur DataFrame
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()

    with open('./output/df_info_log.txt', 'w', encoding='utf-8') as f:
        f.write(info_str)

    st.success("✅ Data berhasil disimpan ke './output/preprocessed_dataset_detik.csv'")

    return df

