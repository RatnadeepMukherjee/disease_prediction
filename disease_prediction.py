import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict on test set
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy on test set: {accuracy:.2f}")

def predict_disease(symptoms):
    """
    Predict disease based on symptoms list.
    Args:
        symptoms (list of str): List of symptoms.
    Returns:
        str: Predicted disease.
    """
    input_data = mlb.transform([symptoms])
    prediction = clf.predict(input_data)
    return prediction[0]

if __name__ == "__main__":
    # Example usage
    user_symptoms = ['fever', 'cough']
    predicted = predict_disease(user_symptoms)
    print(f"Predicted disease for symptoms {user_symptoms}: {predicted}")
