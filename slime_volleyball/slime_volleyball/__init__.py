from maelstrom.population import GeneticProgrammingPopulation
import matplotlib.pyplot as plt


def gather_data(
    right_pop: GeneticProgrammingPopulation,
    left_pop: GeneticProgrammingPopulation,
    evals=None,
) -> dict:
    """
    Gather data from the population
    """
    if right_pop.population is None or left_pop.population is None:
        raise ValueError("Pops must not be None")
    log = {}
    right_fitness = [i.fitness for i in right_pop.population]
    log["avg_right_fitness"] = sum(right_fitness) / len(right_fitness)
    log["max_right_fitness"] = max(right_fitness)
    left_fitness = [i.fitness for i in left_pop.population]
    log["avg_left_fitness"] = sum(left_fitness) / len(left_fitness)
    log["max_left_fitness"] = max(left_fitness)
    if evals:
        log["evals"] = evals
    return log


def plot_data(data: list[dict], title: str, y_label: str, x_label: str):
    """
    Plots the data from the log
    """
    fig, ax = plt.subplots()
    ax.plot(
        [i for i, _ in enumerate(data["avg_right_fitness"])], data["avg_right_fitness"]
    )
    ax.plot(
        [i for i, _ in enumerate(data["avg_left_fitness"])], data["avg_left_fitness"]
    )
    ax.set(xlabel=x_label, ylabel=y_label, title=title)
    plt.show()
