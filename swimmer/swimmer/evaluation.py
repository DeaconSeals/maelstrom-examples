from maelstrom.population import GeneticProgrammingPopulation
from swimmer import gather_data


def evaluation(swimmers: GeneticProgrammingPopulation, **kwargs):
    env = kwargs["env"]
    if swimmers.population is None:
        raise ValueError("Population is empty.")
    for individual in swimmers.population:
        individual.fitness = 0
        observation, info = env.reset()

        for _ in range(1_000):
            action = individual.execute(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            individual.fitness += reward

            if terminated or truncated:
                break
    try:
        swimmers.best.append(max(swimmers.population, key=lambda x: x.fitness))
    except Exception:
        swimmers.best = [max(swimmers.population, key=lambda x: x.fitness)]

    return gather_data(swimmers), len(swimmers.population)
