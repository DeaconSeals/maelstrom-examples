"""Execute the swimmer example"""
import gymnasium as gym
from maelstrom.island import GeneticProgrammingIsland
from snake_eyes import read_config
from swimmer import plot_data
from swimmer.primitives import *
from swimmer.evaluation import evaluation


def main():
    env = gym.make("Swimmer-v4", ctrl_cost_weight=0.1)
    config = read_config("./configs/main.cfg", globals(), locals())
    island = GeneticProgrammingIsland(**config["ISLAND"], **config)
    island.populations["swimmers"].best = []
    island.run()

    # find the best champion and render
    print(island.populations["swimmers"])
    best = max(island.populations["swimmers"].best, key=lambda x: x.fitness)
    genotype = best.genotype
    env.unwrapped.render_mode = "human"
    observation, info = env.reset()
    for _ in range(1_000):
        action = genotype.execute(observation)
        observation, _, terminated, truncated, info = env.step(action)
        env.render()

        if terminated or truncated:
            break
    env.close()

    plot_data(island.log, "Swimmer", "Fitness", "Generation")


if __name__ == "__main__":
    main()
