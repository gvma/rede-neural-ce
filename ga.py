import numpy as np

# UM INDIVIDUO REPRESENTA OS PESOS USADOS NA REDE NEURAL

POPULATION_SIZE = 40

def get_initial_population(X):
  initial_population = []
  for _ in range(POPULATION_SIZE):
    initial_population.append(np.random.uniform(-1, 1, X.shape[1] + 1))
  return initial_population

def score(clf, population):
  scores = []
  for individual in population:
    scores.append(clf.predict(individual))
  return scores

# TODO fazer o crossover entre os individuos
def crossover():
  pass

# TODO fazer a mutacao em um individuo
def mutation():
  pass