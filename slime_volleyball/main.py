"""driver"""
from time import sleep
import gym
import matplotlib.pyplot as plt
from maelstrom.individual import GeneticProgrammingIndividual
from numpy import random
from slimevolleygym import gym
from snake_eyes import read_config
from maelstrom.population import GeneticProgrammingPopulation, statistics
from maelstrom import Maelstrom
from slime_volleyball.primitives import *


def main():
    """
    Main function
    """
    env = gym.make("SlimeVolley-v0")
    config = read_config("./configs/main.cfg", globals(), locals())
    maelstrom = Maelstrom(**config["MAELSTROM"], **config)
    maelstrom = maelstrom.run()
    print(maelstrom.log)
    env.close()
    # plot the results
    for run in maelstrom.log.values():
        plot_data(run, "Maelstrom results", "Generation", "Fitness")


def fitness(
    right_slime: GeneticProgrammingPopulation,
    left_slime: GeneticProgrammingPopulation,
    samples: int = 5,
    **kwargs,
):
    """
    Fitness function for the population
    """
    env = kwargs["env"]
    if left_slime.population is None or right_slime.population is None:
        raise ValueError("Pops must not be None")
    matches = []
    evals = 0
    matches_counter = [0 for _ in right_slime.population]
    for left in range(len(left_slime.population)):
        right_opponents = set()
        for _ in range(samples):
            right_opponent = random.choice(
                [
                    ind
                    for ind in range(len(right_slime.population))
                    if ind not in right_opponents and matches_counter[ind] < 50
                ]
            )
            matches.append((right_opponent, left))
            right_opponents.add(right_opponent)
            matches_counter[right_opponent] += 1

    matches_counter = [0 for _ in left_slime.population]
    for right in range(len(right_slime.population)):
        left_opponents = set()
        for _ in range(samples):
            left_opponent = random.choice(
                [
                    ind
                    for ind in range(len(left_slime.population))
                    if ind not in left_opponents and matches_counter[ind] < 50
                ]
            )
            # matches always follow the format (right, left)
            matches.append((right, left_opponent))
            left_opponents.add(left_opponent)
            matches_counter[left_opponent] += 1
    evaluate_matches(
        right_slime.population,
        left_slime.population,
        matches,
        env,
    )
    right_slime.evals += len(matches)
    left_slime.evals += len(matches)
    evals += len(matches)

    # play a show match between the best individuals
    # find the member of the population with the highest fitness
    best_right = max(right_slime.population, key=lambda x: x.fitness)
    best_left = max(left_slime.population, key=lambda x: x.fitness)

    results = play_match(env, best_right, best_left, render=True)
    env.reset()
    print("Contender 1: ", best_right.fitness)
    print("Contender 2: ", best_left.fitness)
    print("Show match: ", results)

    print("average for this generation:")
    print("right: ", statistics.mean([i.fitness for i in right_slime.population]))
    print("left: ", statistics.mean([i.fitness for i in left_slime.population]))

    return gather_data(right_slime, left_slime, evals), evals


def evaluate_matches(
    right_pop: list[GeneticProgrammingIndividual],
    left_pop: list[GeneticProgrammingIndividual],
    matches: list[tuple[int, int]],
    env,
):
    """
    Evaluates all matches

    Arguments:
        pop1: Population 1
        pop2: Population 2
        matches: List of matches to be played
    """
    for individual in right_pop:
        individual.trials = []
    for individual in left_pop:
        individual.trials = []
    for match in matches:
        right_opp = right_pop[match[0]]
        left_opp = left_pop[match[1]]
        # right_reward, left_reward = play_match(env, right_opp, left_opp)
        reward = play_match(env, right_opp, left_opp)
        env.reset()

        # right_opp.trials.append(right_reward)
        # left_opp.trials.append(left_reward)
        right_opp.trials.append(reward)
        left_opp.trials.append(-reward)
    for individual in right_pop:
        individual.fitness = statistics.mean(individual.trials)
    for individual in left_pop:
        individual.fitness = statistics.mean(individual.trials)


def play_match(
    env,
    right_opp: GeneticProgrammingIndividual,
    left_opp: GeneticProgrammingIndividual,
    render=False,
) -> int:
    # right_reward = 0
    # left_reward = 0
    total_reward = 0
    right_obs = env.reset()
    left_obs = right_obs
    if right_opp.genotype is None or left_opp.genotype is None:
        raise ValueError("Genotype must not be None")
    for rounds_kept_up in range(1000000):
        right_q_values = []
        left_q_values = []
        for action in range(3):
            right_q_values.append(
                right_opp.genotype.execute({"obs": right_obs, "action": action})
            )
            left_q_values.append(
                left_opp.genotype.execute({"obs": left_obs, "action": action})
            )

        right_action = right_q_values.index(max(right_q_values))
        left_action = left_q_values.index(max(left_q_values))

        # turn the actions into a one-hot vector
        right_action = [1 if i == right_action else 0 for i in range(3)]
        left_action = [1 if i == left_action else 0 for i in range(3)]

        right_obs, reward, done, info = env.step(right_action, left_action)
        left_obs = info["otherObs"]
        # right_reward += 1
        # left_reward += 1
        # if reward == 1:
        # right_reward += 20
        # left_reward -= 20
        # elif reward == -1:
        #     right_reward -= 20
        #     left_reward += 20
        total_reward += reward

        if render:
            sleep(0.02)
            env.render()
        if done:
            # if reward == 1:
            #     right_reward += 50
            #     left_reward -= 50
            # elif reward == -1:
            #     right_reward -= 50
            #     left_reward += 50
            break
    # return right_reward, left_reward
    return total_reward


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


if __name__ == "__main__":
    main()
