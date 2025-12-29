import pickle
import pandas as pd

# Charger modèle et encodeurs
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('label_encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

# Simuler un profil qui DEVRAIT avoir un compte
test_profile = {
    'country': 'Kenya',
    'year': 2018,
    'location_type': 'Urban',
    'cellphone_access': 'Yes',
    'household_size': 3,
    'age_of_respondent': 35,
    'gender_of_respondent': 'Male',
    'relationship_with_head': 'Head of Household',
    'marital_status': 'Married/Living together',
    'education_level': 'Tertiary education',
    'job_type': 'Formally employed Private'
}

# Créer DataFrame
df = pd.DataFrame([test_profile])

print("=== AVANT ENCODAGE ===")
print(df.T)
print()

# Encoder
for col in df.columns:
    if col in encoders:
        df[col] = encoders[col].transform(df[col])

print("=== APRÈS ENCODAGE ===")
print(df.T)
print()

# Vérifier l'ordre des features
print("=== FEATURES ATTENDUES PAR LE MODÈLE ===")
if hasattr(model, 'feature_names_in_'):
    print(model.feature_names_in_)
else:
    print("Pas d'info sur les features")
print()

# Prédiction
pred = model.predict(df)
proba = model.predict_proba(df)

print("=== RÉSULTAT ===")
print(f"Prédiction: {'OUI' if pred[0] == 1 else 'NON'}")
print(f"Probabilité NO: {proba[0][0]*100:.1f}%")
print(f"Probabilité YES: {proba[0][1]*100:.1f}%")
