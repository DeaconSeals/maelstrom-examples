import gym
from maelstrom.individual import GeneticProgrammingIndividual
from numpy import random
from slimevolleygym import SlimeVolleyEnv, gym
from snake_eyes import read_config
from maelstrom.population import GeneticProgrammingPopulation, statistics
from maelstrom import Maelstrom
from slime_volleyball.primitives import *
from time import sleep


def main():
    """
    Main function
    """
    env = gym.make("SlimeVolley-v0")
    config = read_config("./configs/main.cfg", globals(), locals())
    maelstrom = Maelstrom(**config["MAELSTROM"], **config)
    maelstrom = maelstrom.run()
    print(maelstrom.log)
    env.reset()
    env.close()


def fitness(
    left_slime: GeneticProgrammingPopulation,
    right_slime: GeneticProgrammingPopulation,
    samples: int = 5,
    **kwargs,
):
    """
    Fitness function for the population
    """
    env = kwargs.get("env")
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
    matches_counter = [0 for _ in left_slime.population]
    for ind1 in range(len(right_slime.population)):
        opponents = set()
        for _ in range(samples):
            opponent = random.choice(
                [
                    ind
                    for ind in range(len(left_slime.population))
                    if ind not in opponents and matches_counter[ind] < 50
                ]
            )
            matches.append((opponent, ind1))
            opponents.add(opponent)
            matches_counter[opponent] += 1
    evaluate_matches(
        left_slime.population,
        right_slime.population,
        matches,
        env,
    )
    left_slime.evals += len(matches)
    right_slime.evals += len(matches)
    evals += len(matches)

    print(
        "Average left fitness: ",
        statistics.mean([i.fitness for i in left_slime.population]),
    )
    print(
        "Average right fitness: ",
        statistics.mean([i.fitness for i in right_slime.population]),
    )

    # show match
    best_left = max(left_slime.population, key=lambda x: x.fitness)
    best_right = max(right_slime.population, key=lambda x: x.fitness)
    res = play_match(env, best_left, best_right, render=True)
    print("Best left fitness: ", best_left.fitness)
    print("Best right fitness: ", best_right.fitness)
    print("Match result: ", res)

    return gather_data(left_slime, right_slime, evals), evals


def evaluate_matches(
    pop1: list[GeneticProgrammingIndividual],
    pop2: list[GeneticProgrammingIndividual],
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
    for individual in pop1:
        individual.trials = []
    for individual in pop2:
        individual.trials = []

    for match in matches:
        opp1 = pop1[match[0]]
        opp2 = pop2[match[1]]
        reward = play_match(env, opp1, opp2)
        opp1.trials.append(reward)
        opp2.trials.append(reward)

    for individual in pop1:
        individual.fitness = statistics.mean(individual.trials)
    for individual in pop2:
        individual.fitness = statistics.mean(individual.trials)


def play_match(env, opp1, opp2, render=False):
    total_reward = 0
    obs1 = env.reset()
    obs2 = obs1
    if opp1.genotype is None or opp2.genotype is None:
        raise ValueError("Genotype must not be None")
    for kept_up in range(10000):
        q_values1 = []
        q_values2 = []
        for action in range(3):
            q_values1.append(opp1.genotype.execute({"obs": obs1, "action": action}))
            q_values2.append(opp2.genotype.execute({"obs": obs2, "action": action}))

        action1 = q_values1.index(max(q_values1))
        action2 = q_values2.index(max(q_values2))

        # transform the action into a one-hot vector
        action1 = [1 if i == action1 else 0 for i in range(3)]
        action2 = [1 if i == action2 else 0 for i in range(3)]

        obs1, reward, done, info = env.step(action1, action2)
        obs2 = info["otherObs"]
        total_reward = kept_up
        if render:
            sleep(0.02)
            env.render()
        if done:
            break
    return total_reward


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
