from maelstrom.population import GeneticProgrammingPopulation
from mountain_car import gather_data


def evaluation(cars: GeneticProgrammingPopulation, **kwargs):
    env = kwargs["env"]
    if cars.population is None:
        raise ValueError("Population is empty.")
    for individual in cars.population:
        individual.fitness = 0
        if individual.genotype is None:
            raise ValueError("Individual has no genotype.")
        observation, info = env.reset()

        for _ in range(1_000):
            q_values = []
            for i in range(3):
                q_values.append(
                    individual.genotype.execute({"obs": observation, "action": i})
                )
            action = q_values.index(max(q_values))
            observation, reward, terminated, truncated, info = env.step(action)
            individual.fitness += reward

            if terminated or truncated:
                break
    try:
        cars.best.append(max(cars.population, key=lambda x: x.fitness))
    except Exception:
        cars.best = [max(cars.population, key=lambda x: x.fitness)]

    return gather_data(cars), len(cars.population)
