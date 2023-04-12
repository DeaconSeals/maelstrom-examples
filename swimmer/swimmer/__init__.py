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
    # first, write data to file
    print(data)
    with open("data.txt", "w") as f:
        # data in the format of {"avg_fitness":[], "max_fitness":[]}
        # one column for generation number, which corresponds to the index
        # one column for the average fitness
        # one column for the max fitness

        # write the header
        f.write(f"{x_label}\t|\t{y_label}\t|\tMax Fitness\n")
        for i, _ in enumerate(data["avg_fitness"]):
            f.write(f"{i}\t|\t{data['avg_fitness'][i]}\t|\t{data['max_fitness'][i]}\n")

    fig, ax = plt.subplots()
    ax.set_ylim([-500, 500])
    ax.plot([i for i, _ in enumerate(data["avg_fitness"])], data["avg_fitness"])
    ax.set(xlabel=x_label, ylabel=y_label, title=title)
    plt.show()
