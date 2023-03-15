from maelstrom.genotype import GeneticTree
from maelstrom.individual import GeneticProgrammingIndividual
from maelstrom.population import GeneticProgrammingPopulation

import random
import math
import statistics
from collections import OrderedDict


# General-purpose GP population class that contains and manages individuals
# TODO: transition from parameters dictionary to clearer inputs with default values
class CanonicalGeneticProgrammingPopulation(GeneticProgrammingPopulation):
    def __init__(
        self,
        pop_size,
        roles,
        output_type,
        depth_limit,
        hard_limit=None,
        depth_min=1,
        evaluations=None,
        parent_selection="uniform",
        mutation=0.05,
        reproduction=0.0,
        **kwargs
    ):
        self.population = list()
        # self.parameters = parameters
        self.pop_size = pop_size
        self.roles = roles
        self.output_type = output_type
        self.depth_limit = depth_limit
        self.hard_limit = hard_limit if hard_limit is not None else self.depth_limit * 2
        self.depth_min = depth_min
        self.eval_limit = evaluations
        self.evals = 0
        self.parent_selection = parent_selection
        self.mutation = mutation
        self.reproduction = reproduction
        self.optional_params = kwargs
        self.hall_of_fame = OrderedDict()
        self.CIAO = []

    # Generate children through the selection of parents, recombination or mutation of parents to form children, then the migration of children
    # into the primary population depending on survival strategy
    # TODO: generalize this so it relies on operations of the individual class instead of skipping that and working directly with the genotype
    def generate_children(self, imports=None):
        if imports != None:
            children = [migrant.copy() for migrant in imports]
        else:
            children = []

        copied = set()

        while len(children) < self.pop_size:
            prob = random.random()
            if prob <= self.mutation:
                children.append(self.select_parents(1)[0].copy())
                children[-1].genotype.subtree_mutation()
            elif prob <= self.mutation + self.reproduction:
                parent = self.select_parents(1)[0]
                if parent.genotype.string in copied:
                    continue
                copied.add(parent.genotype.string)
                children.append(parent.copy())
            else:
                parents = self.select_parents(2)
                children.append(parents[0].copy())
                children[-1].genotype.subtree_recombination(parents[1].genotype)

        self.population = children
