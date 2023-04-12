"""driver"""
import gym
from slimevolleygym import gym
from snake_eyes import read_config
from maelstrom import Maelstrom
from slime_volleyball import plot_data
from slime_volleyball.fitness.basic import basic_fitness
from slime_volleyball.fitness.mix import mix_fitness
from slime_volleyball.fitness.volley import volley_fitness
from slime_volleyball.primitives import *


def main():
    """
    Main function
    """
    env = gym.make("SlimeVolley-v0")
    # config = read_config("./configs/basic.cfg", globals(), locals())
    # config = read_config("./configs/mix.cfg", globals(), locals())
    # config = read_config("./configs/volley.cfg", globals(), locals())
    config = read_config("./configs/maelstrom.cfg", globals(), locals())
    maelstrom = Maelstrom(**config["MAELSTROM"], **config)
    maelstrom = maelstrom.run()
    print(maelstrom.log)
    env.close()
    # plot the results
    for run in maelstrom.log.values():
        plot_data(run, "Maelstrom results", "Generation", "Fitness")


if __name__ == "__main__":
    main()
