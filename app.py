import streamlit as st
import pandas as pd
import pickle

# Configuration de la page
st.set_page_config(page_title="Prédiction Inclusion Financière", page_icon="💳", layout="centered")

# CSS personnalisé pour un thème bleu nuit
st.markdown("""
    <style>
        .stApp {
            background-color: #0f1c2e;
            color: white;
        }
        h1, h2, h3, h4, h5, h6, p, label {
            color: white;
        }
        .stButton>button {
            background-color: #29587e;
            color: white;
            border: 1px solid white;
            width: 100%;
        }
        .stButton>button:hover {
            background-color: #3a6ea5;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Titre
st.title("💳 Prédiction d'Inclusion Financière")

# Chargement du modèle et des encodeurs
@st.cache_resource
def load_model():
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
        return model, label_encoders
    except Exception as e:
        st.error(f"❌ Erreur de chargement : {e}")
        return None, None

model, label_encoders = load_model()

if model is not None and label_encoders is not None:
    # Formulaire
    st.markdown("---")
    
    # Pays
    country = st.selectbox("Pays", ["Kenya", "Rwanda", "Tanzania", "Uganda"])
    
    # Année
    year = st.selectbox("Année", [2016, 2017, 2018])
    
    # Type de lieu
    location_type = st.selectbox("Type de lieu", ["Rural", "Urban"])
    
    # Accès au téléphone portable
    cellphone_access = st.selectbox("Accès au téléphone portable", ["Yes", "No"])
    
    # Taille du ménage
    household_size = st.number_input("Taille du ménage", min_value=1, max_value=21, value=3)
    
    # Âge
    age_of_respondent = st.slider("Âge du répondant", min_value=16, max_value=100, value=30)
    
    # Genre
    gender_of_respondent = st.selectbox("Genre", ["Male", "Female"])
    
    # Relation avec le chef de ménage
    relationship_with_head = st.selectbox(
        "Lien avec le chef de ménage",
        ["Head of Household", "Spouse", "Child", "Parent", "Other relative", "Other non-relatives"]
    )
    
    # Statut matrimonial
    marital_status = st.selectbox(
        "Statut matrimonial",
        ["Married/Living together", "Widowed", "Single/Never Married", "Divorced/Seperated", "Dont know"]
    )
    
    # Niveau d'éducation
    education_level = st.selectbox(
        "Niveau d'éducation",
        ["Secondary education", "No formal education", "Vocational/Specialised training", 
         "Primary education", "Tertiary education", "Other/Dont know/RTA"]
    )
    
    # Type d'emploi
    job_type = st.selectbox(
        "Statut professionnel",
        ["Self employed", "Government Dependent", "Formally employed Private", 
         "Informally employed", "Formally employed Government", "Farming and Fishing",
         "Remittance Dependent", "Other Income", "Dont Know/Refuse to answer", "No Income"]
    )
    
    # Bouton de prédiction
    if st.button("Prédire"):
        try:
            # Créer le dataframe avec les données
            input_data = pd.DataFrame({
                'country': [country],
                'year': [year],
                'location_type': [location_type],
                'cellphone_access': [cellphone_access],
                'household_size': [household_size],
                'age_of_respondent': [age_of_respondent],
                'gender_of_respondent': [gender_of_respondent],
                'relationship_with_head': [relationship_with_head],
                'marital_status': [marital_status],
                'education_level': [education_level],
                'job_type': [job_type]
            })
            
            # Encoder les variables catégorielles
            for col in input_data.columns:
                if col in label_encoders:
                    try:
                        input_data[col] = label_encoders[col].transform(input_data[col])
                    except ValueError as e:
                        st.warning(f"⚠️ Valeur non reconnue pour {col}: {input_data[col].values[0]}")
                        # Utiliser la première classe connue comme fallback
                        input_data[col] = 0
            
            # Faire la prédiction
            prediction = model.predict(input_data)
            prediction_proba = model.predict_proba(input_data)
            
            # Afficher le résultat
            st.markdown("---")
            # Logique normale: 0=No, 1=Yes
            if prediction[0] == 1:
                st.success(f"✅ **A un compte bancaire** (Confiance: {prediction_proba[0][1]*100:.1f}%)")
            else:
                st.error(f"❌ **N'a pas de compte bancaire** (Confiance: {prediction_proba[0][0]*100:.1f}%)")
                
        except Exception as e:
            st.error(f"❌ Erreur lors de la prédiction : {e}")
else:
    st.error("❌ Impossible de charger le modèle. Vérifiez que les fichiers model.pkl et label_encoders.pkl sont présents.")



