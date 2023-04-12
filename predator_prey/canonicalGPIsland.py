from canonicalPopulation import CanonicalGeneticProgrammingPopulation
from maelstrom.island import GeneticProgrammingIsland
from tqdm.auto import tqdm
import multiprocessing


# General-purpose island class that contains and manages multiple populations
# TODO: transition from parameters dictionary to clearer inputs with default values
class CanonicalGeneticProgrammingIsland(GeneticProgrammingIsland):
    # Initializes the island and populations based on input configuration parameters and evaluation function
    def __init__(
        self,
        populations,
        evaluation_function,
        evaluationkwargs=dict(),
        eval_pool=None,
        evaluations=None,
        champions_per_generation=0,
        cores=None,
        position=None,
        **kwargs
    ):
        # self.parameters = parameters
        self.populations = {}
        self.generation_count = 0
        for name, config in populations.items():
            self.populations[name] = CanonicalGeneticProgrammingPopulation(
                **kwargs[config]
            )
            self.populations[name].ramped_half_and_half()
        self.evaluation = evaluation_function

        self.evaluation_parameters = evaluationkwargs

        self.log = {}
        # if evalPool is None:
        # 	if cores is None:
        # 		cores = min(32, multiprocessing.cpu_count())
        # 	self.evalPool = multiprocessing.Pool(cores)
        # else:
        # 	self.evalPool = evalPool

        if cores is None:
            cores = min(32, multiprocessing.cpu_count())
        self.cores = cores
        self.position = position

        # Fitness evaluations occur here
        with multiprocessing.Pool(self.cores) as eval_pool:
            generation_data, self.evals = self.evaluation(
                **self.populations, executor=eval_pool, **self.evaluation_parameters
            )
        for key in generation_data:
            self.log[key] = [generation_data[key]]

        self.champions_per_generation = champions_per_generation

        # identify champions for each species
        self.champions = {key: {} for key in self.populations}
        for population in self.populations:
            local_champions = self.select(
                population, self.champions_per_generation, method="best"
            )
            for individual in local_champions:
                gene_text = individual.genotype.print_tree()
                if gene_text not in self.champions[population]:
                    self.champions[population][gene_text] = individual.genotype.copy()

        self.imports = {}
        self.eval_limit = evaluations

    # Performs a single generation of evolution
    def generation(self, eval_pool=None):
        self.generation_count += 1
        for population in self.populations:
            if population in self.imports:
                self.populations[population].generate_children(self.imports[population])
            else:
                self.populations[population].generate_children()
        self.imports.clear()

        generation_data, num_evals = self.evaluation(
            **self.populations, executor=eval_pool, **self.evaluation_parameters
        )
        self.evals += num_evals
        for key in generation_data:
            self.log[key].append(generation_data[key])

        for population in self.populations:
            self.populations[population].update_hall_of_fame()

            # identify champions for each species
            local_champions = self.select(
                population, self.champions_per_generation, method="best"
            )
            for individual in local_champions:
                gene_text = individual.genotype.print_tree()
                if geneText not in self.champions[population]:
                    self.champions[population][gene_text] = individual.genotype.copy()

        return self
