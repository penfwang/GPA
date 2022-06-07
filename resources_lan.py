import random
from deap import tools


def varAnd(population, toolbox, cxpb, mutpb):
    """Part of an evolutionary algorithm applying only the variation part
    (crossover **and** mutation). The modified individuals have their
    fitness invalidated. The individuals are cloned so returned population is
    independent of the input population.

    :param population: A list of individuals to vary.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :returns: A list of varied individuals that are independent of their
              parents.

    The variation goes as follow. First, the parental population
    :math:`P_\mathrm{p}` is duplicated using the :meth:`toolbox.clone` method
    and the result is put into the offspring population :math:`P_\mathrm{o}`.  A
    first loop over :math:`P_\mathrm{o}` is executed to mate pairs of
    consecutive individuals. According to the crossover probability *cxpb*, the
    individuals :math:`\mathbf{x}_i` and :math:`\mathbf{x}_{i+1}` are mated
    using the :meth:`toolbox.mate` method. The resulting children
    :math:`\mathbf{y}_i` and :math:`\mathbf{y}_{i+1}` replace their respective
    parents in :math:`P_\mathrm{o}`. A second loop over the resulting
    :math:`P_\mathrm{o}` is executed to mutate every individual with a
    probability *mutpb*. When an individual is mutated it replaces its not
    mutated version in :math:`P_\mathrm{o}`. The resulting :math:`P_\mathrm{o}`
    is returned.

    This variation is named *And* because of its propensity to apply both
    crossover and mutation on the individuals. Note that both operators are
    not applied systematically, the resulting individuals can be generated from
    crossover only, mutation only, crossover and mutation, and reproduction
    according to the given probabilities. Both probabilities should be in
    :math:`[0, 1]`.
    """
    offspring = [toolbox.clone(ind) for ind in population]

    # Apply crossover and mutation on the offspring
    for i in range(1, len(offspring), 2):
        if random.random() < cxpb:
            offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1],
                                                          offspring[i])
            del offspring[i - 1].fitness.values, offspring[i].fitness.values

    for i in range(len(offspring)):
        if random.random() < mutpb:
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values

    return offspring


def eaSimple(population, toolbox, cxpb, mutpb, elitismNum, ngen, stats=None,halloffame=None, verbose=__debug__):

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)
    # Begin the generational process

    record = stats.compile(population) if stats else {}
    # print(record)
    logbook.record(gen=0, nevals=len(population), **record)
    if verbose:
        print(logbook.stream)

    hof2 = tools.HallOfFame(1)
    # offspring_for_va = toolbox.selectElitism(population, k=1)
    # hof2 = evalValidation(offspring_for_va, toolbox, hof2)
    # Begin the generational process
    # best_ind_va=[]
    # best_ind_va.append(hof2[0])

    for gen in range(1, ngen + 1):
        offspring_for_va = toolbox.selectElitism(population, k=1)
        hof2 = evalValidation(offspring_for_va, toolbox, hof2)
        # best_ind_va.append(offspring_for_va[0])
        # print(best_ind_va[-1],best_ind_va[-1].fitness)
        # print(gen, 'evaluate validation')
        # print(gen, 'hof1', halloffame[0].fitness, halloffame[0])
        # print(gen, 'hof2', hof2[0].fitness, hof2[0])

        # Select the next generation individuals by elitism
        offspringE = toolbox.selectElitism(population, k=elitismNum)
        # print(len(offspringE))
        # Select the next generation individuals for crossover and mutation
        offspring = toolbox.select(population, len(population) - elitismNum)
        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)

        # add offspring from elitism into current offspring
        # generate the next generation individuals
        offspring[0:0] = offspringE

        # offspring_for_va[:] = offspring
        # Evaluate the individuals with an invalid fitness
        # invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, offspring)
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit
            # print('fit',fit)

        # Update the hall of fame with the generated
        if halloffame is not None:
            halloffame.update(offspring)
        # best_int_train.append(halloffame[0])
        # update the hall of fame for validation set
        # update population
        population[:] = offspring
        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(offspring), **record)
        # print(record)
        if verbose:
            print(logbook.stream)

    return population, logbook, hof2


def evalValidation(offspring_for_va, toolbox, hof2):
    fitnesses2 = toolbox.map(toolbox.validation, offspring_for_va)
    for ind2, fit2 in zip(offspring_for_va, fitnesses2):
        ind2.fitness.values = fit2
    if hof2 is not None:
        hof2.update(offspring_for_va)
    return hof2
