import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils import shuffle
import os


class DiseasePredictionModel:
    def __init__(self):
        self.model = None
        self.symptom_weights = {}
        self.diseases = []

    def load_and_preprocess_data(self, dataset_path, severity_path):
        """Load and preprocess the dataset"""
        # Load main dataset
        df = pd.read_csv(dataset_path)
        df = shuffle(df, random_state=42)

        # Remove underscores, clean whitespace and lowercase
        for col in df.columns:
            df[col] = df[col].astype(str).str.replace('_', ' ', regex=False) \
                            .str.replace(r'\s+', ' ', regex=True) \
                            .str.strip().str.lower()

        # Flatten and trim remaining whitespaces
        cols = df.columns
        data = df[cols].values.flatten()
        s = pd.Series(data)
        s = s.str.strip()
        s = s.values.reshape(df.shape)
        df = pd.DataFrame(s, columns=df.columns)

        # Fill NaN values with 0
        df = df.fillna(0)

        # Load symptom severity data
        severity_df = pd.read_csv(severity_path)
        severity_df['Symptom'] = severity_df['Symptom'].astype(str).str.replace('_', ' ', regex=False) \
                                                .str.replace(r'\s+', ' ', regex=True) \
                                                .str.strip().str.lower()

        # Create symptom weight mapping
        for _, row in severity_df.iterrows():
            self.symptom_weights[row['Symptom']] = row['weight']

        # Encode symptoms with weights
        vals = df.values
        symptoms = severity_df['Symptom'].unique()

        for i in range(len(symptoms)):
            vals[vals == symptoms[i]] = severity_df[severity_df['Symptom'] == symptoms[i]]['weight'].values[0]

        encoded_df = pd.DataFrame(vals, columns=cols)

        # ✅ FIX: Only replace text with 0 in feature columns, not 'Disease'
        feature_cols = encoded_df.columns.drop('Disease')
        encoded_df[feature_cols] = encoded_df[feature_cols].replace(to_replace=r'[a-z ]+', value=0, regex=True)

        return encoded_df

    def train_model(self, dataset_path, severity_path):
        """Train the disease prediction model"""
        print("Loading and preprocessing data...")
        df = self.load_and_preprocess_data(dataset_path, severity_path)

        # Prepare features and labels
        X = df.iloc[:, 1:].values.astype(float)
        y = df['Disease'].values

        self.diseases = list(df['Disease'].unique())
        print("All unique diseases in the dataset:", np.unique(y))
        print("Disease distribution in y:", pd.Series(y).value_counts())

        # Stratified split
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in sss.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

        print("Disease distribution in y_train:", pd.Series(y_train).value_counts())

        print("Training models...")
        # Initialize models
        rf_model = RandomForestClassifier(
            random_state=42,
            max_features='sqrt',
            n_estimators=500,
            max_depth=13
        )
        dt_model = DecisionTreeClassifier(
            criterion='gini',
            random_state=42,
            max_depth=13
        )
        svm_model = SVC(probability=True, random_state=42)  # probability=True required for predict_proba

        # Create ensemble model
        self.model = VotingClassifier(
            estimators=[
                ('rf', rf_model),
                ('dt', dt_model),
                ('svm', svm_model)
            ],
            voting='soft'
        )

        # Train the model
        self.model.fit(X_train, y_train)

        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')

        print(f"Model Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")

        # Cross-validation
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=kfold, scoring='accuracy')
        print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        return accuracy, f1

    def predict_disease(self, symptoms):
        """Predict disease based on symptoms"""
        if self.model is None:
            raise ValueError("Model not trained yet!")

        feature_vector = self.symptoms_to_vector(symptoms)
        prediction = self.model.predict([feature_vector])
        probability = self.model.predict_proba([feature_vector])
        return prediction[0], max(probability[0])

    def symptoms_to_vector(self, symptoms):
        """Convert list of symptoms to feature vector"""
        feature_vector = [0] * 17
        for i, symptom in enumerate(symptoms[:17]):
            if symptom and symptom.strip() and symptom.strip().lower() in self.symptom_weights:
                feature_vector[i] = self.symptom_weights[symptom.strip().lower()]
        return feature_vector

    def save_model(self, filepath):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'symptom_weights': self.symptom_weights,
            'diseases': self.diseases
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.symptom_weights = model_data['symptom_weights']
        self.diseases = model_data['diseases']
        print(f"Model loaded from {filepath}")


# Training script
if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    disease_model = DiseasePredictionModel()

    try:
        accuracy, f1 = disease_model.train_model('data/dataset.csv', 'data/Symptom-severity.csv')
        disease_model.save_model('models/disease_model.joblib')

        print("Model training completed successfully!")
        print(f"Final Accuracy: {accuracy:.4f}")
        print(f"Final F1 Score: {f1:.4f}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure the data files are in the 'data/' directory")
