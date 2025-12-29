import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

print("🔄 Chargement des données...")
df = pd.read_csv('Financial_inclusion_dataset.csv')

# Afficher la distribution des classes
print(f"\n📊 Distribution originale:")
print(df['bank_account'].value_counts())
print(f"Proportion Yes: {(df['bank_account']=='Yes').sum()/len(df)*100:.1f}%")

# Sélectionner les features
features = ['country', 'year', 'location_type', 'cellphone_access', 
            'household_size', 'age_of_respondent', 'gender_of_respondent',
            'relationship_with_head', 'marital_status', 'education_level', 'job_type']

X = df[features].copy()
y = df['bank_account'].copy()

# Encoder la target CORRECTEMENT (0=No, 1=Yes)
print("\n🔧 Encodage de la target...")
y = y.map({'No': 0, 'Yes': 1})
print(f"Classes: 0=No, 1=Yes")

# Encoder les features catégorielles
print("\n🔧 Encodage des features...")
label_encoders = {}
for col in X.columns:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\n📊 Distribution train:")
print(f"  No: {(y_train==0).sum()} ({(y_train==0).sum()/len(y_train)*100:.1f}%)")
print(f"  Yes: {(y_train==1).sum()} ({(y_train==1).sum()/len(y_train)*100:.1f}%)")

# Entraîner le modèle avec class_weight balanced
print("\n🤖 Entraînement du Random Forest...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced',  # IMPORTANT: équilibrer les classes
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Évaluation
print("\n📊 Évaluation sur le test set...")
y_pred = model.predict(X_test)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.2%}")
print("\nRapport de classification:")
print(classification_report(y_test, y_pred, target_names=['No', 'Yes']))

# Sauvegarder
print("\n💾 Sauvegarde du modèle et des encodeurs...")
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

print("\n✅ Modèle réentraîné et sauvegardé avec succès!")
print("\n🧪 Test avec un profil favorable:")
test_profile = pd.DataFrame([{
    'country': 0,  # Kenya
    'year': 2018,
    'location_type': 1,  # Urban
    'cellphone_access': 1,  # Yes
    'household_size': 3,
    'age_of_respondent': 35,
    'gender_of_respondent': 1,  # Male
    'relationship_with_head': 1,  # Head of Household
    'marital_status': 2,  # Married
    'education_level': 4,  # Tertiary
    'job_type': 3  # Formally employed Private
}])

pred = model.predict(test_profile)
proba = model.predict_proba(test_profile)

print(f"Prédiction: {'✅ YES' if pred[0] == 1 else '❌ NO'}")
print(f"Probabilité Yes: {proba[0][1]*100:.1f}%")
