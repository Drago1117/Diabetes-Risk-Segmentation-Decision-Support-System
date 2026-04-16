import pandas as pd
import joblib
import os

class DiabetesPredictor:
    def __init__(self):
        self.model = joblib.load('models/xgb_classifier.pkl')
        self.le = joblib.load('models/label_encoder.pkl')
        self.feature_names = joblib.load('models/class_features.pkl')
    
    def preprocess(self, input_data):
        # input_data is dict with keys like Age, Gender (0/1), BMI, etc.
        df = pd.DataFrame([input_data])
        # Drop Name if present
        if 'Name' in df:
            df = df.drop('Name', axis=1)
        # Reorder/drop to match training, add dummies
        df = df.reindex(columns=self.feature_names, fill_value=0)
        df = pd.get_dummies(df, drop_first=True)
        # Align to training features
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[self.feature_names]
        return df
    
    def predict(self, input_data):
        X = self.preprocess(input_data)
        pred_enc = self.model.predict(X)[0]
        pred_class = self.le.inverse_transform([pred_enc])[0]
        prob = self.model.predict_proba(X)[0]
        return {'risk_stage': pred_class, 'probabilities': dict(zip(self.le.classes_, prob))}

predictor = DiabetesPredictor()

def predict_risk(input_data):
    """input_data dict of patient features"""
    return predictor.predict(input_data)

