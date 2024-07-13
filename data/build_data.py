import pandas as pd
from model import Model
from data.eda import EDA

#Arquivo
csv_path = 'diabetes.csv'

def build_and_train_model(csv_path):
    data = pd.read_csv(csv_path)
    #VIEW
    diabetes_model = Model(data)
    diabetes_model.view_data()

    #EDA
    eda = EDA(data)
    eda.view_distribute()
    eda.view_relations()

    #TEST
    diabetes_model.processing_data()
    diabetes_model.train_model()
    eval_results = diabetes_model.evaluate_model()
    
    #BUILD
    print("Acurácia:", eval_results["accuracy_score"])
    print("Relatório de Classificação:\n", eval_results["classification_report"])
    print("Matriz de Confusão:\n", eval_results["confusion_matrix"])
    return diabetes_model