"""Execute the mountain car example"""
import gymnasium as gym
from maelstrom.island import GeneticProgrammingIsland
from snake_eyes import read_config
from mountain_car import plot_data
from mountain_car.primitives import *
from mountain_car.evaluation import evaluation


def main():
    env = gym.make("MountainCar-v0")
    config = read_config("./configs/main.cfg", globals(), locals())
    island = GeneticProgrammingIsland(**config["ISLAND"], **config)
    island.populations["cars"].best = []
    island.run()

    plot_data(island.log, "Car", "Fitness", "Generation")

    # find the best champion and render
    best = max(island.populations["cars"].best, key=lambda x: x.fitness)
    print("Best agent's fitness: ", best.fitness)
    genotype = best.genotype
    env.unwrapped.render_mode = "human"
    observation, _ = env.reset()
    fitness = 0
    for _ in range(1_000):
        q_values = []
        for i in range(3):
            q_values.append(genotype.execute({"obs": observation, "action": i}))
        action = q_values.index(max(q_values))
        observation, reward, terminated, truncated, _ = env.step(action)
        env.render()


if __name__ == "__main__":
    main()
