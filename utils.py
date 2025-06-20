import streamlit as st
import pandas as pd
import ast
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

def convert_string_to_list(text):
    return ast.literal_eval(text)
