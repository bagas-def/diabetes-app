import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==================== KONFIGURASI ====================
st.set_page_config(page_title="Deteksi Risiko Diabetes", page_icon="🩺", layout="centered")

# ==================== INISIALISASI RIWAYAT ====================
if "riwayat" not in st.session_state:
    st.session_state.riwayat = []

# ==================== LOAD MODEL & DATASET ====================
with open('model_diabetes.pkl', 'rb') as file:
    model = pickle.load(file)

data = pd.read_csv("diabetes.csv")
data.columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

# ==================== MODE TAMPILAN ====================
mode = st.sidebar.selectbox("🎨 Mode Tampilan", ["Terang", "Gelap"])
if mode == "Gelap":
    st.markdown("""
        <style>
        body { background-color: #121212; color: white; }
        .stApp { background-color: #121212; }
        </style>
    """, unsafe_allow_html=True)

# ==================== NAVIGASI ====================
halaman = st.sidebar.radio("Navigasi", ["🏠 Prediksi", "ℹ️ Tentang Aplikasi"])

# ==================== HALAMAN PREDIKSI ====================
if halaman == "🏠 Prediksi":
    st.title("🩺 Aplikasi Deteksi Risiko Diabetes")
    st.markdown('<div class="big-font">Masukkan data pasien di sidebar untuk memprediksi risiko terkena diabetes berdasarkan data medis menggunakan algoritma <b>Random Forest</b>.</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.subheader("📌 Statistik Ringkas Dataset")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🔌 Glukosa (Rata-rata)", f"{data['Glucose'].mean():.1f}")
        st.metric("🩸 Tekanan Darah (Rata-rata)", f"{data['BloodPressure'].mean():.1f}")
        st.metric("📏 BMI (Rata-rata)", f"{data['BMI'].mean():.1f}")
        st.metric("💉 Insulin (Rata-rata)", f"{data['Insulin'].mean():.1f}")
    with col2:
        st.metric("👩 Kehamilan (Rata-rata)", f"{data['Pregnancies'].mean():.1f}")
        st.metric("📈 Usia (Rata-rata)", f"{data['Age'].mean():.1f}")
        st.metric("📊 DPF (Rata-rata)", f"{data['DiabetesPedigreeFunction'].mean():.2f}")
        st.metric("📋 Total Data", len(data))

    st.subheader("📊 Distribusi Nilai Medis (Histogram)")
    fitur_pilih = st.selectbox("Pilih fitur:", data.columns[:-1])
    fig_hist, ax_hist = plt.subplots()
    sns.histplot(data[fitur_pilih], kde=True, color="skyblue", ax=ax_hist)
    ax_hist.set_title(f"Distribusi Nilai {fitur_pilih}")
    st.pyplot(fig_hist)

    st.markdown("---")

    st.sidebar.title("📋 Form Input Data Pasien")
    Pregnancies = st.sidebar.number_input("Jumlah Kehamilan", 0, 20, 1)
    Glucose = st.sidebar.number_input("Glukosa", 0, 200, 120)
    BloodPressure = st.sidebar.number_input("Tekanan Darah", 0, 140, 70)
    SkinThickness = st.sidebar.number_input("Tebal Lipatan Kulit", 0, 100, 20)
    Insulin = st.sidebar.number_input("Insulin", 0, 900, 80)
    BMI = st.sidebar.number_input("BMI", 0.0, 70.0, 25.0)
    DPF = st.sidebar.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    Age = st.sidebar.number_input("Usia", 1, 120, 30)

    st.subheader("📈 Hasil Prediksi Risiko Diabetes")
    if st.button("🔍 Prediksi"):
        input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                                Insulin, BMI, DPF, Age]])
        hasil = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        st.session_state.riwayat.append({
            "Glukosa": Glucose, "BMI": BMI, "Insulin": Insulin, "Usia": Age,
            "Hasil": "Risiko" if hasil == 1 else "Tidak Risiko",
            "Probabilitas": f"{prob:.2%}"
        })

        if hasil == 1:
            st.error(f"⚠️ Hasil: **Pasien berisiko diabetes.**\n\nProbabilitas: `{prob:.2%}`")
        else:
            st.success(f"✅ Hasil: **Pasien tidak berisiko diabetes.**\n\nProbabilitas: `{prob:.2%}`")

        st.info(" 💡 **Saran:**\n\n- Jaga pola makan sehat\n- Periksa kadar gula darah secara rutin\n- Konsultasikan ke dokter untuk diagnosa lanjutan")
        if hasattr(model, 'feature_importances_'):
            st.subheader("🔬 Pengaruh Fitur terhadap Prediksi")
            fitur = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                     'Insulin', 'BMI', 'DPF', 'Age']
            importance = model.feature_importances_

            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(x=importance, y=fitur, palette="rocket", ax=ax)
            ax.set_title("Tingkat Pengaruh Setiap Fitur")
            ax.set_xlabel("Importance")
            ax.set_xlim(0, max(importance) + 0.05)
            for i, v in enumerate(importance):
                ax.text(v + 0.005, i, f"{v:.2f}", va='center')
            st.pyplot(fig)

        st.subheader("📓 Riwayat Prediksi")
        df = pd.DataFrame(st.session_state.riwayat)
        st.dataframe(df)
        if not df.empty:
            csv = df.to_csv(index=False).encode()
            st.download_button("📅 Unduh Hasil Prediksi", csv, "riwayat_prediksi.csv", "text/csv")

# ==================== HALAMAN TENTANG ====================
elif halaman == "ℹ️ Tentang Aplikasi":
    st.header("ℹ️ Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini digunakan untuk memprediksi risiko diabetes berdasarkan data medis pasien menggunakan algoritma **Random Forest**.

    **Fitur yang digunakan:**
    - Jumlah Kehamilan
    - Glukosa
    - Tekanan Darah
    - Tebal Lipatan Kulit
    - Insulin
    - BMI
    - Diabetes Pedigree Function
    - Usia

    **Dataset**: Pima Indian Diabetes Dataset  
    **Model**: Random Forest Classifier  
    **Akurasi**: Sekitar 75%

    Dikembangkan oleh: **Bagas Syifa Pratama (2025)**
    """)

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("<center><i>Dikembangkan oleh Bagas Syifa Pratama · 2025</i></center>", unsafe_allow_html=True)
