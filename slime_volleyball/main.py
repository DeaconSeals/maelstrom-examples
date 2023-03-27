import gym
from maelstrom.individual import GeneticProgrammingIndividual
from numpy import random
from slimevolleygym import SlimeVolleyEnv, gym
from snake_eyes import read_config
from maelstrom.population import GeneticProgrammingPopulation
from maelstrom import Maelstrom
from slime_volleyball.primitives import *


def main():
    """
    Main function
    """
    config = read_config("./configs/main.cfg", globals(), locals())
    maelstrom = Maelstrom(**config["MAELSTROM"], **config)
    maelstrom = maelstrom.run()
    print(maelstrom.log)


def fitness(
    left_slime: GeneticProgrammingPopulation,
    right_slime: GeneticProgrammingPopulation,
    samples: int = 5,
    **kwargs,
):
    """
    Fitness function for the population
    """
    if left_slime.population is None or right_slime.population is None:
        raise ValueError("Pops must not be None")
    matches = []
    evals = 0
    matches_counter = [0 for _ in right_slime.population]
    for ind1 in range(len(left_slime.population)):
        opponents = set()
        for _ in range(samples):
            opponent = random.choice(
                [
                    ind
                    for ind in range(len(right_slime.population))
                    if ind not in opponents and matches_counter[ind] < 50
                ]
            )
            matches.append((ind1, opponent))
            opponents.add(opponent)
            matches_counter[opponent] += 1
    evaluate_matches(
        left_slime.population,
        right_slime.population,
        matches,
    )
    left_slime.evals += len(matches)
    right_slime.evals += len(matches)
    evals += len(matches)
    return gather_data(left_slime, right_slime, evals), evals


def evaluate_matches(
    pop1: list[GeneticProgrammingIndividual],
    pop2: list[GeneticProgrammingIndividual],
    matches: list[tuple[int, int]],
):
    """
    Evaluates all matches

    Arguments:
        pop1: Population 1
        pop2: Population 2
        matches: List of matches to be played
    """
    results = {}
    # env = gym.make("SlimeVolley-v0")
    env = SlimeVolleyEnv()

    for match in matches:
        total_reward = 0
        obs1 = env.reset()
        obs2 = obs1
        opp1 = pop1[match[0]]
        opp2 = pop2[match[1]]
        if opp1.genotype is None or opp2.genotype is None:
            raise ValueError("Genotype must not be None")
        for _ in range(1000):
            q_values1 = []
            q_values2 = []
            for action in range(3):
                q_values1.append(opp1.genotype.execute({"obs": obs1, "action": action}))
                q_values2.append(opp2.genotype.execute({"obs": obs2, "action": action}))

            action1 = q_values1.index(max(q_values1))
            action2 = q_values2.index(max(q_values2))
            if action1 == 0:
                action1 = [1, 0, 0]
            if action1 == 1:
                action1 = [0, 1, 0]
            if action1 == 2:
                action1 = [0, 0, 1]
            if action2 == 0:
                action2 = [1, 0, 0]
            if action2 == 1:
                action2 = [0, 1, 0]
            if action2 == 2:
                action2 = [0, 0, 1]

            obs1, reward, done, info = env.step(action1, action2)
            obs2 = info["otherObs"]
            total_reward += reward
            env.render()
            if done:
                break
        env.close()
        if results.get(match[0]):
            results[match[0]] = (results[match[0]] + total_reward) / 2
        else:
            results[match[0]] = total_reward
        if results.get(match[1]):
            results[match[1]] = (results[match[1]] + -total_reward) / 2
        else:
            results[match[1]] = -total_reward
    for index, individual in enumerate(pop1):
        individual.fitness = results[index]
    for index, individual in enumerate(pop2):
        individual.fitness = results[index]


def gather_data(
    pop1: GeneticProgrammingPopulation, pop2: GeneticProgrammingPopulation, evals=None
) -> dict:
    """
    Gather data from the population
    """
    if pop1.population is None or pop2.population is None:
        raise ValueError("Pops must not be None")
    log = {}
    pop1_fitness = [i.fitness for i in pop1.population]
    log["avg_pop1_fitness"] = sum(pop1_fitness) / len(pop1_fitness)
    log["max_pop1_fitness"] = max(pop1_fitness)
    pop2_fitness = [i.fitness for i in pop2.population]
    log["avg_pop2_fitness"] = sum(pop2_fitness) / len(pop2_fitness)
    log["max_pop2_fitness"] = max(pop2_fitness)
    if evals:
        log["evals"] = evals
    return log


if __name__ == "__main__":
    main()
