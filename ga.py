import numpy as np
from numpy import random

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

def exchange_dna(father, mother, new_population):
  first_child = father[:1] + mother[1:]
  second_child = mother[:1] + father[1:]
  new_population.append(first_child)
  new_population.append(second_child)
  return new_population

def crossover(new_population, population):
  for i in range(len(population)):
    father_index = np.random.randint(0, len(population)-1)
    mother_index = np.random.randint(0, len(population)-1)
    while father_index == mother_index:
      father_index = np.random.randint(0, len(population)-1)
      mother_index = np.random.randint(0, len(population)-1)
    father = population[father_index]
    mother = population[mother_index]
    new_population = exchange_dna(father, mother, new_population)
  return new_population

def mutation(population):
  for i in range(2):
    index = np.random.randint(0, len(population) - 1)
    individual = population[index]
    position1 = np.random.randint(0,3)
    position2 = position1
    while position1 == position2:
      position2 = np.random.randint(0,3)
    aux = individual[position1]
    individual[position1] = individual[position2]
    individual[position2] = aux