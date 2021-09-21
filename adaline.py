import numpy as np
import pandas as pd

class Adaline(object):
    def __init__(self, eta = 0.001, epoch = 100):
      self.eta = eta
      self.epoch = epoch
      self.df = pd.read_csv('Dados_Treinamento_Sinal.csv',header=None)
      self.df.head()

      self.X = self.df.iloc[0:35,[0,1,2,3]].values
      self.y = self.df.iloc[0:35,4].values

    def net_input(self, individual, weight_):
      return np.dot(individual, weight_[1:]) + weight_[0]
    def activation_function(self, individual, weight_):
      return self.net_input(individual, weight_)
    def predict(self, weight_):
      score = 0
      for index, data in self.df.iterrows():
        individual = []
        answer = None
        for i, v in data.items():
          if i != 4:
            individual.append(v)
          else:
            answer = v
        prediction = np.where(self.activation_function(individual, weight_) >= 0.0, 1, -1)
        if prediction == answer:
          score += 1
      return score