# Alunos
# Guilherme Volney Mota Amaral
# João Victor Ferro
# Arthur Bernardo Sávio de Melo

from ga import *
from adaline import *

clf = Adaline()

initial_population = get_initial_population(clf.X)
scores = score(clf, initial_population)
scores.sort(reverse=True)
population = initial_population

while True:
  if scores[0] == 35:
    break
  new_population = crossover([], population)
  mutation(new_population)
  new_score = score(clf, new_population)
  if new_score[0] == 35:
    print('Pesos: ', new_score[0])
    break
  break
  # population = selection(population, new_population)