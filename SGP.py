import math
import operator
import pandas as pd
import numpy as np
from deap import creator, base, tools, gp
import resources_lan
import sys,saveFile
import math,time
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import random
from sklearn import preprocessing
toolbox = base.Toolbox()

def protectedDiv(left, right):
    if right == 0:
        return 0
    else:
        return left/right


def normalise(data,test_data):
    for i in range(data.shape[1]):
      column = data[:,i]
      mini = min(column)
      maxi = max(column)
      if mini != maxi:
         for j in range(len(column)):
            data[j,i] = (column[j] - mini) / (maxi - mini)
         test_column = test_data[:, i]
         for jj in range(len(test_column)):
           test_data[jj,i] = (test_column[i] - mini) / (maxi - mini)
    return data,test_data


def fitness_function(individual,training_data,training_labels,unique_label):
    func = toolbox.compile(expr=individual)
    outputs = []
    for instance in training_data:
        predicted_label = func(*instance)
        if predicted_label >= 0:
            outputs.append(min(unique_label))
        else:
            outputs.append(max(unique_label))
    right = 0
    for i in range(len(training_labels)):
        if outputs[i] == training_labels[i]:
            right = right + 1
    acc = right/len(training_labels)
    return (1-acc),


def main(seed,dataset_name):
 ##################################loading the data
    folder1 = '/nesi/project/vuw03334/split_GP' + '/' + 'train' + str(dataset_name) + ".npy"
    x_train = np.load(folder1)
    
    
    folder2 = '/nesi/project/vuw03334/split_GP' + '/' + 'validation' + str(dataset_name) + ".npy"
    x_validation = np.load(folder2)
    

    label_to_see = list(set(x_train[:,0]))
    training_data = preprocessing.normalize(x_train[:, 1:])
    validation_data = preprocessing.normalize(x_validation[:, 1:])
    
    
 ##################################loading the data
 ##################################using all features
    feature_number = len(training_data[0])
    if feature_number < 300:
        MU = feature_number  ####the number of particle
    else:
        MU = 300  #####bound to 300
 ##################################using all features
    pset = gp.PrimitiveSet("MAIN", feature_number)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protectedDiv, 2)
    # pset.addPrimitive(operator.neg, 1)
    # pset.addPrimitive(square, 1)
    # pset.addPrimitive(maxi, 2)
    # pset.addPrimitive(math.cos, 1)
    # pset.addPrimitive(math.sin, 1)
    # for i in range(-10, 10):
    #     pset.addTerminal(i)

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=3, max_=8)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    
    toolbox.register("selectElitism", tools.selBest)
    toolbox.register("evaluate", fitness_function, training_data=training_data,training_labels=x_train[:, 0],unique_label= label_to_see)
    toolbox.register("validation", fitness_function, training_data=validation_data, training_labels=x_validation[:, 0],unique_label=label_to_see)
    toolbox.register("select", tools.selTournament, tournsize=7)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=3)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=6))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=6))
    toolbox.decorate("expr_mut", gp.staticLimit(key=operator.attrgetter("height"), max_value=6))

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)
    hof = tools.HallOfFame(1)
    pop = toolbox.population(n=MU)
 ### cxpb: The probability of mating two individuals;
 # mutpb: The probability of mutating an individual;
 # ngen: The number of generation.
    pop, log, hof2 = resources_lan.eaSimple(pop, toolbox, cxpb=0.8, mutpb=0.2, elitismNum=1, ngen=100, stats=mstats, halloffame=hof, verbose=True)
    function = toolbox.compile(expr=hof[0])
    # print(hof[0])
    return pop,hof2[0]


if __name__ == "__main__":
    dataset_name = str(sys.argv[1])
    seed = str(sys.argv[2])
    #dataset_name = 'dataSet_ion'
    #seed = str(1)
    random.seed(int(seed))
    start = time.time()
    pop,p_one= main(seed, dataset_name)
    end = time.time()
    running_time = end - start
    saveFile.save_individual(seed, dataset_name, p_one)
    saveFile.saveAllfeature7(seed, dataset_name, running_time)

