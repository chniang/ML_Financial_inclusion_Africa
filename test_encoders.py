import pickle
import pandas as pd

# Charger les encodeurs
with open('label_encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

# Afficher les classes connues pour chaque encodeur
print("=== CLASSES CONNUES PAR LES ENCODEURS ===\n")
for col, encoder in encoders.items():
    print(f"{col}:")
    print(f"  Classes: {list(encoder.classes_)}")
    print()
