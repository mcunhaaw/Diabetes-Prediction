import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

csv_path = 'diabetes.csv'

class Model:
    def __init__(self, data):
        self.data = data
        self.scaler = StandardScaler()
        self.model = LogisticRegression()
        
    def view_data(self):
        print("Tabela de classificações")
        print(self.data.head())

    def processing_data(self):
        X = self.data.drop('Resultado', axis=1)
        y = self.data['Resultado']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    def train_model(self):
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        return {
        "accuracy_score": accuracy_score(self.y_test, y_pred),  
        "classification_report": classification_report(self.y_test, y_pred), 
        "confusion_matrix": confusion_matrix(self.y_test, y_pred) 
    }

    def predict(self, input_data):
        input_data = self.scaler.transform([input_data])
        return 