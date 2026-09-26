import streamlit as st
import pandas as pd
import numpy as np
import joblib


# =========================
# Load Model dan Scaler
# =========================

kmeans = joblib.load("kmeans_apartment.pkl")
scaler = joblib.load("scaler_apartment.pkl")

cluster_profile = pd.read_csv(
    "cluster_profile.csv",
    index_col=0
)

cluster_profile.index = cluster_profile.index.astype(int)


# =========================
# Konfigurasi Halaman
# =========================

st.set_page_config(
    page_title="Apartment Clustering",
    page_icon="🏠",
    layout="centered"
)


# =========================
# Judul Aplikasi
# =========================

st.title("🏠 Apartment Clustering")

st.write(
    "Aplikasi untuk mengelompokkan apartemen berdasarkan "
    "harga, luas, jumlah bedrooms, dan jumlah bathrooms "
    "menggunakan algoritma K-Means."
)


# =========================
# Input Data Apartemen
# =========================

st.subheader("Input Data Apartemen")

price = st.number_input(
    "Harga Sewa per Bulan ($)",
    min_value=200.0,
    value=1200.0,
    step=50.0
)

square_feet = st.number_input(
    "Luas Apartemen (square feet)",
    min_value=1.0,
    value=800.0,
    step=50.0
)

bedrooms = st.number_input(
    "Jumlah Bedrooms",
    min_value=0.0,
    value=1.0,
    step=1.0
)

bathrooms = st.number_input(
    "Jumlah Bathrooms",
    min_value=0.5,
    value=1.0,
    step=0.5
)


# =========================
# Prediksi Cluster
# =========================

if st.button("🔍 Prediksi Cluster"):

    # Transformasi log
    price_log = np.log1p(price)
    square_feet_log = np.log1p(square_feet)

    # Data input sesuai urutan fitur training
    input_data = pd.DataFrame({
        "price_log": [price_log],
        "square_feet_log": [square_feet_log],
        "bedrooms": [bedrooms],
        "bathrooms": [bathrooms]
    })

    # Standardisasi menggunakan scaler hasil training
    input_scaled = scaler.transform(input_data)

    # Prediksi cluster
    cluster = kmeans.predict(input_scaled)[0]

    # =========================
    # Hasil Clustering
    # =========================

    st.success(
        f"🏠 Apartemen termasuk **Cluster {cluster}**"
    )

    st.subheader("Profil Cluster")

    profile = cluster_profile.loc[cluster]

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Jumlah Apartemen",
            f"{int(profile['jumlah_data']):,}"
        )

        st.metric(
            "Median Price",
            f"${profile['median_price']:,.0f}"
        )

        st.metric(
            "Median Square Feet",
            f"{profile['median_square_feet']:,.0f} sq ft"
        )

    with col2:

        st.metric(
            "Median Bedrooms",
            f"{profile['median_bedrooms']:.0f}"
        )

        st.metric(
            "Median Bathrooms",
            f"{profile['median_bathrooms']:.1f}"
        )

    st.info(
        "Cluster menunjukkan kelompok apartemen berdasarkan "
        "kemiripan karakteristik harga, luas, bedrooms, dan bathrooms."
    )