import matplotlib.pyplot as plt
from maelstrom.population import GeneticProgrammingPopulation


def gather_data(
    swimmers: GeneticProgrammingPopulation,
    evals=None,
) -> dict:
    """
    Gather data from the population
    """
    if swimmers.population is None:
        raise ValueError("Pops must not be None")
    log = {}
    fitness = [i.fitness for i in swimmers.population]
    log["avg_fitness"] = sum(fitness) / len(fitness)
    log["max_fitness"] = max(fitness)
    if evals:
        log["evals"] = evals
    return log


def plot_data(data: list[dict], title: str, y_label: str, x_label: str):
    """
    Plots the data from the log
    """
    fig, ax = plt.subplots()
    ax.plot([i for i, _ in enumerate(data["avg_fitness"])], data["avg_fitness"])
    ax.set(xlabel=x_label, ylabel=y_label, title=title)
    plt.show()
