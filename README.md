# 📊 ML Inclusion Financière Afrique

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ml-financial-inclusion-africa.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Prédiction de l'accès aux services bancaires en Afrique de l'Est avec Machine Learning**

Application web interactive permettant de prédire si un individu a accès à un compte bancaire en fonction de ses caractéristiques socio-économiques. Projet basé sur des données réelles de 4 pays d'Afrique de l'Est : Kenya, Rwanda, Tanzanie et Ouganda.

---

## 🎯 Contexte & Objectif

L'**inclusion financière** est un enjeu majeur en Afrique. Ce projet vise à :

- 🔍 **Analyser** les facteurs socio-économiques influençant l'accès aux services bancaires
- 🤖 **Prédire** l'accès bancaire à partir de 13 variables démographiques et économiques
- 📊 **Visualiser** les insights pour aider à cibler les campagnes d'inclusion financière
- 🌍 **Contribuer** à réduire l'exclusion financière en Afrique de l'Est

---

## 📈 Dataset

- **Source** : Financial Inclusion in Africa Dataset
- **Observations** : 23,524 individus
- **Pays couverts** : Kenya (59%), Rwanda (20%), Tanzanie (14%), Ouganda (7%)
- **Période** : 2016-2018
- **Variables** : 13 features (âge, genre, éducation, emploi, localisation, etc.)
- **Target** : `bank_account` (binaire : Oui/Non)

### Variables clés

| Variable | Description | Type |
|----------|-------------|------|
| `country` | Pays de résidence | Catégorielle |
| `year` | Année d'enquête | Numérique |
| `location_type` | Type de zone (Rural/Urban) | Catégorielle |
| `cellphone_access` | Accès au téléphone portable | Binaire |
| `household_size` | Taille du ménage | Numérique |
| `age_of_respondent` | Âge du répondant | Numérique |
| `gender_of_respondent` | Genre | Catégorielle |
| `relationship_with_head` | Lien avec le chef de ménage | Catégorielle |
| `marital_status` | Statut matrimonial | Catégorielle |
| `education_level` | Niveau d'éducation | Catégorielle |
| `job_type` | Type d'emploi | Catégorielle |
| **`bank_account`** | **A un compte bancaire (cible)** | **Binaire** |

---

## 🛠️ Technologies Utilisées

### Data Science & ML
- **Python 3.11** - Langage principal
- **Pandas** - Manipulation de données
- **NumPy** - Calculs numériques
- **Scikit-learn** - Modèle Random Forest
- **YData Profiling** - Analyse exploratoire automatisée

### Visualisation & Déploiement
- **Matplotlib & Seaborn** - Visualisations statistiques
- **Streamlit** - Application web interactive
- **Streamlit Cloud** - Hébergement gratuit

---

## 🚀 Installation & Utilisation

### Option 1 : Utiliser l'application en ligne (recommandé)

👉 **[Lancer l'application](https://ml-financial-inclusion-africa.streamlit.app)**

### Option 2 : Installation locale

#### Prérequis
- Python 3.11+
- pip

#### Étapes

1. **Cloner le repository**
```bash
git clone https://github.com/chniang/ML_Financial_inclusion_Africa.git
cd ML_Financial_inclusion_Africa
```

2. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

3. **Lancer l'application**
```bash
streamlit run app.py
```

4. **Accéder à l'app**
```
Ouvrir http://localhost:8501 dans votre navigateur
```

---

## 📊 Fonctionnalités

### 1️⃣ Analyse Exploratoire (EDA)

Le notebook `Financial_inclusion_dataset.ipynb` contient :

- ✅ Nettoyage et préparation des données
- ✅ Statistiques descriptives complètes
- ✅ Visualisations des distributions
- ✅ Analyse des corrélations
- ✅ Rapport automatisé YData Profiling

### 2️⃣ Modèle de Prédiction

- **Algorithme** : Random Forest Classifier
- **Features** : 11 variables encodées
- **Encodage** : LabelEncoder pour variables catégorielles
- **Sauvegarde** : Modèle et encodeurs sérialisés (pickle)

### 3️⃣ Application Interactive

Interface Streamlit permettant de :

- 🎯 Saisir les caractéristiques d'un individu via formulaire
- 🤖 Obtenir une prédiction instantanée (A un compte / N'a pas de compte)
- 📊 Visualiser les insights clés du dataset

---

## 🔬 Méthodologie

### 1. Collecte & Nettoyage
- Import du dataset CSV
- Vérification des valeurs manquantes (0%)
- Vérification des doublons (0%)

### 2. Analyse Exploratoire
- Génération automatique du rapport avec YData Profiling
- Analyse des distributions par variable
- Étude des corrélations
- Identification des facteurs discriminants

### 3. Préparation des Données
- Encodage des variables catégorielles (LabelEncoder)
- Séparation features/target
- Split train/test

### 4. Modélisation
- Entraînement d'un Random Forest
- Sauvegarde du modèle et des encodeurs
- Intégration dans l'application Streamlit

### 5. Déploiement
- Développement de l'interface utilisateur
- Tests en local
- Déploiement sur Streamlit Cloud

---

## 📌 Insights Clés

D'après l'analyse exploratoire :

- 📱 **76%** des personnes ayant accès au téléphone portable ont un compte bancaire
- 🏙️ **68%** des résidents urbains ont un compte vs **42%** en zone rurale
- 🎓 Le **niveau d'éducation** est fortement corrélé à l'inclusion bancaire
- 💼 Le **type d'emploi** est un facteur déterminant
- 👨‍👩‍👧‍👦 Les **chefs de famille** ont plus souvent un compte bancaire

---

## 📁 Structure du Projet
```
ML_Financial_inclusion_Africa/
├── app.py                              # Application Streamlit
├── Financial_inclusion_dataset.csv     # Dataset brut
├── Financial_inclusion_dataset.ipynb   # Notebook d'analyse
├── model.pkl                           # Modèle Random Forest entraîné
├── label_encoders.pkl                  # Encodeurs pour les variables
├── rapport_profilage.html              # Rapport YData Profiling
├── requirements.txt                    # Dépendances Python
└── README.md                           # Ce fichier
```

---

## 🎓 Auteur

**Cheikh Niang**  
Data Scientist | Machine Learning Engineer

- 📧 Email: [cheikhniang159@gmail.com](mailto:cheikhniang159@gmail.com)
- 💼 LinkedIn: [linkedin.com/in/cheikh-niang-5370091b5](https://www.linkedin.com/in/cheikh-niang-5370091b5/)
- 💻 GitHub: [github.com/chniang](https://github.com/chniang)

---

## 📜 Licence

Ce projet est libre d'utilisation à des fins éducatives ou personnelles. Toute reproduction commerciale nécessite une autorisation préalable.

---

## 🙏 Remerciements

- Dataset fourni par des initiatives d'inclusion financière en Afrique
- Communauté Streamlit pour le support technique
- Inspiré par les enjeux d'accessibilité bancaire en Afrique de l'Est

---

## 🔮 Améliorations Futures

- [ ] Ajouter d'autres algorithmes ML (XGBoost, LightGBM)
- [ ] Implémenter le feature importance
- [ ] Créer un dashboard de monitoring
- [ ] Ajouter des explications avec SHAP
- [ ] Développer une API REST

---

<div align="center">

**⭐ N'hésitez pas à mettre une étoile si ce projet vous a été utile ! ⭐**

</div>
