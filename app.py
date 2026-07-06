import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("model_svm_kelulusan.joblib")
scaler = joblib.load("scaler.joblib")
selector = joblib.load("selector.joblib")

st.set_page_config(page_title="Prediksi Kelulusan Mahasiswa", layout="centered")

st.sidebar.title("📚 Tentang Aplikasi")

st.sidebar.info("""
Aplikasi ini dibuat untuk memprediksi
kelulusan mahasiswa menggunakan
algoritma Support Vector Machine (SVM).

Dataset:
Higher Education Student Dataset

Model:
Support Vector Machine (SVM)
""")

st.title("🎓 Sistem Prediksi Kelulusan Mahasiswa")

st.markdown("""
Silakan masukkan data mahasiswa pada form di bawah ini.
Kemudian tekan tombol **Prediksi** untuk mengetahui hasil prediksi
menggunakan algoritma **Support Vector Machine (SVM)**.
""")

# =====================
# INPUT USER
# =====================

st.subheader("📝 Input Data Mahasiswa")
course = st.number_input("Course", value=0)

tuition = st.selectbox(
    "Tuition fees up to date",
    [0, 1]
)

scholarship = st.selectbox(
    "Scholarship holder",
    [0, 1]
)

age = st.number_input(
    "Age at enrollment",
    value=18
)

eval1 = st.number_input(
    "Curricular units 1st sem (evaluations)",
    value=0
)

approved1 = st.number_input(
    "Curricular units 1st sem (approved)",
    value=0
)

grade1 = st.number_input(
    "Curricular units 1st sem (grade)",
    value=0.0
)

eval2 = st.number_input(
    "Curricular units 2nd sem (evaluations)",
    value=0
)

approved2 = st.number_input(
    "Curricular units 2nd sem (approved)",
    value=0
)

grade2 = st.number_input(
    "Curricular units 2nd sem (grade)",
    value=0.0
)

# =====================
# DATAFRAME
# =====================

input_data = pd.DataFrame([{
    "Course": course,
    "Tuition fees up to date": tuition,
    "Scholarship holder": scholarship,
    "Age at enrollment": age,
    "Curricular units 1st sem (evaluations)": eval1,
    "Curricular units 1st sem (approved)": approved1,
    "Curricular units 1st sem (grade)": grade1,
    "Curricular units 2nd sem (evaluations)": eval2,
    "Curricular units 2nd sem (approved)": approved2,
    "Curricular units 2nd sem (grade)": grade2
}])

if st.button("🔍 Prediksi Kelulusan"):

    input_scaled = scaler.transform(input_data)

    input_selected = selector.transform(input_scaled)

    hasil = model.predict(input_selected)

    st.divider()

    st.subheader("📊 Hasil Prediksi")

    if hasil[0] == 1:
        st.success("🎓 Mahasiswa Diprediksi LULUS")

    elif hasil[0] == 0:
        st.error("❌ Mahasiswa Diprediksi DROP OUT")

    else:
        st.warning("📚 Mahasiswa Diprediksi ENROLLED")

    st.info("""
Model yang digunakan adalah Support Vector Machine (SVM)
dengan 10 fitur hasil Feature Selection menggunakan SelectKBest.
""")

    st.markdown("---")

st.caption("""
UAS Machine Learning

Nama : Farid Eka Maulana

Program Studi : Teknik Informatika

Algoritma : Support Vector Machine (SVM)
""")
