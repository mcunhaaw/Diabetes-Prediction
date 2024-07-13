import seaborn as sns
import matplotlib.pyplot as plt

class EDA:
    def __init__(self, data):
        self.data = data

    def view_distribute(self):
        sns.countplot(x='Resultado', data=self.data)
        plt.title("Distribuição dos resultados")
        plt.show()

    def view_relations(self):
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.data.corr(), annot=True, cmap='coolwarm')
        plt.title("Correlação")
        plt.show()