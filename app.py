import streamlit as st
import pandas as pd
import pickle


#CSS personnalisé pour un thème bleu nuit
st.markdown("""
    <style>
        /* Couleur de fond globale */
        .stApp {
            background-color: #0f1c2e;
            color: white;
        }

        /* Style des titres et labels */
        h1, h2, h3, h4, h5, h6, p, label, .css-1cpxqw2, .css-1v0mbdj {
            color: white;
        }

        /* Style des champs de saisie */
        input, textarea, select {
            background-color: #1c2e4a !important;
            color: white !important;
            border: 1px solid white !important;
        }

        /* Style des boutons */
        .stButton>button {
            background-color: #29587e;
            color: white;
            border: 1px solid white;
        }

        .stButton>button:hover {
            background-color: #3a6ea5;
            color: white;
        }

        /* Centrer le titre */
        .title {
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# === Chargement du modèle et des encodeurs ===
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("❌ Le fichier 'model.pkl' est introuvable.")

try:
    with open('label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
except FileNotFoundError:
    st.error("❌ Le fichier 'label_encoders.pkl' est introuvable.")

st.title("💳 Prédiction d'Inclusion Financière")

# Champs utilisateur
country = st.selectbox("Pays", ["Kenya", "Rwanda", "Tanzania", "Uganda"])
location_type = st.selectbox("Type de lieu", ["Rural", "Urban"])
cellphone_access = st.selectbox("Accès au téléphone portable", ["Yes", "No"])
household_size = st.number_input("Taille du ménage", min_value=1, step=1)
age = st.slider("Âge du répondant", 10, 100, 30)
gender = st.selectbox("Genre", ["Female", "Male"])
relationship = st.selectbox("Lien avec le chef de ménage", [
    "Head of Household", "Spouse", "Child", "Parent", "Other relative", "Other non-relatives"
])
marital_status = st.selectbox("Statut matrimonial", [
    "Married/Living together", "Single/Never Married", "Widowed", "Divorced/Seperated", "Dont know"
])
education = st.selectbox("Niveau d’éducation", [
    "No formal education", "Primary education", "Secondary education",
    "Tertiary education", "Vocational training", "Other"
])
job = st.selectbox("Statut professionnel", [
    "Self employed", "Government Dependent", "Formally employed Private",
    "Informally employed", "Farming and Fishing", "Formally employed Government",
    "Remittance Dependent", "Other Income"
])

# Valeurs fixes ou générées
year = 2016

# Créer le DataFrame
input_data = pd.DataFrame({
    'country': [country],
    'year': [year],
    'location_type': [location_type],
    'cellphone_access': [cellphone_access],
    'household_size': [household_size],
    'age_of_respondent': [age],
    'gender_of_respondent': [gender],
    'relationship_with_head': [relationship],
    'marital_status': [marital_status],
    'education_level': [education],
    'job_type': [job]
})

# Encodage uniquement des colonnes catégorielles
for col in input_data.columns:
    if col in label_encoders:
        try:
            input_data[col] = label_encoders[col].transform(input_data[col])
        except Exception as e:
            st.warning(f"⚠️ Erreur d'encodage pour la colonne {col} : {e}")

# Réorganiser les colonnes si le modèle les attend dans un ordre particulier
try:
    input_data = input_data[model.feature_names_in_]
except AttributeError:
    pass  # Si modèle sans feature_names_in_

# === Prédiction ===
if st.button("Prédire"):
    try:
        prediction = model.predict(input_data)
        st.success("✅ A un compte bancaire" if prediction[0] == 1 else "❌ N'a pas de compte bancaire")
    except Exception as e:
        st.error(f"Une erreur est survenue lors de la prédiction : {e}")
