import pandas as pd
import joblib

class Clusterer:
    def __init__(self):
        self.kmeans = joblib.load('models/kmeans.pkl')
        self.scaler = joblib.load('models/scaler.pkl')
        self.feature_names = joblib.load('models/seg_features.pkl')
    
    def preprocess(self, input_data):
        df = pd.DataFrame([input_data])
        if 'Name' in df:
            df = df.drop('Name', axis=1)
        df = df.reindex(columns=self.feature_names, fill_value=0)
        df = pd.get_dummies(df, drop_first=True).astype(float)
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[self.feature_names]
        return df
    
    def get_cluster(self, input_data):
        X = self.preprocess(input_data)
        X_scaled = self.scaler.transform(X)
        cluster = self.kmeans.predict(X_scaled)[0]
        return cluster

clusterer = Clusterer()

def get_cluster(input_data):
    return clusterer.get_cluster(input_data)

