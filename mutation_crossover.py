
import random
from collections import defaultdict, deque



# Define the name of type for any types.
__type__ = object

def cxOnePoint(ind1, ind2):
    """Randomly select crossover point in each individual and exchange each
    subtree with the point as root between each individual.

    :param ind1: First tree participating in the crossover.
    :param ind2: Second tree participating in the crossover.
    :returns: A tuple of two trees.
    """
    if len(ind1) < 2 or len(ind2) < 2:
        # No crossover on single node tree
        return ind1, ind2
    # List all available primitive types in each individual
    types1 = defaultdict(list)
    types2 = defaultdict(list)
    if ind1.root.ret == __type__:
        # Not STGP optimization
        types1[__type__] = range(1, len(ind1))
        types2[__type__] = range(1, len(ind2))
        common_types = [__type__]
    else:
        for idx, node in enumerate(ind1[1:], 1):
            types1[node.ret].append(idx)
        for idx, node in enumerate(ind2[1:], 1):
            types2[node.ret].append(idx)
        common_types = set(types1.keys()).intersection(set(types2.keys()))##[<class 'object'>]


    if len(common_types) > 0:
        type_ = random.choice(list(common_types))

        index1 = random.choice(types1[type_])
        index2 = random.choice(types2[type_])
        # print(common_types, random.choice(list(common_types)),index1)
        # ind_names = list(map(lambda x: x.__class__.__name__ if isinstance(x, gp_tree.Ephemeral) else x.name, ind1))
        # print(ind_names)
        # print(len(ind_names),ind_names[index1-1],ind_names[index1],ind_names[index1+1])
        # print(ind1.searchSubtree(index1))
        # exit()

        slice1 = ind1.searchSubtree(index1)
        slice2 = ind2.searchSubtree(index2)
        ind1[slice1], ind2[slice2] = ind2[slice2], ind1[slice1]

    return ind1, ind2




def mutUniform(individual, expr, pset):
    """Randomly select a point in the tree *individual*, then replace the
    subtree at that point as a root by the expression generated using method
    :func:`expr`.

    :param individual: The tree to be mutated.
    :param expr: A function object that can generate an expression when
                 called.
    :returns: A tuple of one tree.
    """
    index = random.randrange(len(individual))# the numer of nodes
    slice_ = individual.searchSubtree(index)
    # print(index,slice_)
    # print(individual[index].ret)
    type_ = individual[index].ret
    individual[slice_] = expr(pset=pset, type_=type_)
    # plot_tree.plot_tree_python_tool(pset)
    return individual,
