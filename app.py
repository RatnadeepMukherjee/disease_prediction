import streamlit as st
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier

# Sample dataset: symptoms and corresponding diseases
data = {
    'symptoms': [
        ['fever', 'cough', 'fatigue'],
        ['headache', 'nausea', 'dizziness'],
        ['fever', 'rash', 'joint pain'],
        ['cough', 'shortness of breath', 'chest pain'],
        ['fatigue', 'weight loss', 'night sweats'],
        ['headache', 'fever', 'stiff neck'],
        ['abdominal pain', 'diarrhea', 'vomiting'],
        ['fever', 'cough', 'sore throat'],
        ['joint pain', 'swelling', 'redness'],
        ['fatigue', 'pale skin', 'shortness of breath']
    ],
    'disease': [
        'Flu',
        'Migraine',
        'Dengue',
        'Pneumonia',
        'Tuberculosis',
        'Meningitis',
        'Food Poisoning',
        'Common Cold',
        'Arthritis',
        'Anemia'
    ]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Binarize symptoms for multi-label classification
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df['symptoms'])

# Target variable
y = df['disease']

# Train classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X, y)

def predict_disease(symptoms):
    input_data = mlb.transform([symptoms])
    prediction = clf.predict(input_data)
    return prediction[0]

# Dictionary mapping diseases to treatment methods
treatment_methods = {
    'Flu': 'Rest, hydration, antiviral medications if prescribed.',
    'Migraine': 'Pain relievers, anti-nausea medications, lifestyle changes.',
    'Dengue': 'Supportive care, hydration, pain relievers (avoid aspirin).',
    'Pneumonia': 'Antibiotics, rest, fluids, hospitalization if severe.',
    'Tuberculosis': 'Long-term antibiotic treatment under medical supervision.',
    'Meningitis': 'Antibiotics or antiviral medications, hospitalization.',
    'Food Poisoning': 'Hydration, rest, sometimes antibiotics.',
    'Common Cold': 'Rest, fluids, over-the-counter cold remedies.',
    'Arthritis': 'Anti-inflammatory drugs, physical therapy, lifestyle changes.',
    'Anemia': 'Iron supplements, diet changes, treatment of underlying cause.'
}

st.title("Disease Prediction Model Based on Symptoms")

st.write("Select symptoms from the list below:")

all_symptoms = sorted(set([symptom for sublist in data['symptoms'] for symptom in sublist]))

selected_symptoms = st.multiselect("Symptoms", all_symptoms)

if st.button("Predict Disease"):
    if not selected_symptoms:
        st.warning("Please select at least one symptom.")
    else:
        disease = predict_disease(selected_symptoms)
        st.success(f"Predicted Disease: {disease}")
        treatment = treatment_methods.get(disease, "Treatment information not available.")
        st.info(f"Suggested Treatment: {treatment}")
