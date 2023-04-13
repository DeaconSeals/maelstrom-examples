from maelstrom import Maelstrom
from maelstrom.island import GeneticProgrammingIsland
from maelstrom.population import GeneticProgrammingPopulation
from primitives import *
from competition import *
from findChampions import localTournaments


import math
import random
import statistics
from pathlib import Path
import gzip
import pickle
import sys
import json
from time import strftime, localtime
import multiprocessing
import concurrent.futures

from snake_eyes import read_config
from argparse import ArgumentParser

# from PIL import Image, ImageDraw # (commented out so dependencies are resolved)
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm.auto import trange
from tqdm.contrib.concurrent import process_map

# Renders an interaction between agents (commented out so dependencies are resolved)
# def render_world(world, name="world", log_path="logs"):
# 	FULL_SIZE = 800
# 	DOWN_SIZE = 400
# 	AGENT_SIZE = world.agent_radius * FULL_SIZE / 2

# 	def world_to_image(x, y):
# 		x = (x + 1) * FULL_SIZE / 2
# 		y = (y + 1) * FULL_SIZE / 2
# 		return x, y

# 	frames = list()
# 	for frame in range(len(world.past_predator_positions)):
# 		image = Image.new("RGB", (FULL_SIZE, FULL_SIZE), (128, 128, 128))
# 		draw = ImageDraw.Draw(image)
# 		draw.ellipse(((0, 0), (FULL_SIZE - 1, FULL_SIZE - 1)), (255, 255, 255), (0, 0, 0, 0), 5)

# 		predator = world_to_image(*world.past_predator_positions[frame])
# 		draw.ellipse(((predator[0] - AGENT_SIZE, predator[1] - AGENT_SIZE),
# 					  (predator[0] + AGENT_SIZE, predator[1] + AGENT_SIZE)), (255, 0, 0), (0, 0, 0), 2)
# 		prey = world_to_image(*world.past_prey_positions[frame])
# 		draw.ellipse(((prey[0] - AGENT_SIZE, prey[1] - AGENT_SIZE), (prey[0] + AGENT_SIZE, prey[1] + AGENT_SIZE)),
# 					 (0, 255, 0), (0, 0, 0), 2)

# 		for i in range(1, frame + 1):
# 			start = world_to_image(*world.past_predator_positions[i - 1])
# 			end = world_to_image(*world.past_predator_positions[i])
# 			draw.line((start, end), (255, 0, 0), 5, "curve")
# 			start = world_to_image(*world.past_prey_positions[i - 1])
# 			end = world_to_image(*world.past_prey_positions[i])
# 			draw.line((start, end), (0, 255, 0), 5, "curve")

# 		image.thumbnail((DOWN_SIZE, DOWN_SIZE))
# 		frames.append(image)
# 	error = PermissionError("No error?")
# 	for _ in range(10):
# 		try:
# 			frames[0].save(log_path + "/{0}.gif".format(name), save_all=True, append_images=frames[1:], optimize=False,
# 						   duration=40, loop=0)
# 			return
# 		except PermissionError as e:
# 			time.sleep(1)
# 			error = e
# 	raise error


def main():
    arg_parser = ArgumentParser(
        description="Maelstrom Predator Prey Experiment Driver",
        epilog="Example: driver.py " "--config configs/default.cfg",
    )
    arg_parser.add_argument(
        "--config",
        type=str,
        default="./configs/default.cfg",
        help="Configuration file for experiment parameters",
    )
    args = arg_parser.parse_args()

    config = read_config(args.config, globals(), locals())
    # print(config.keys())
    # for section in config:
    # 	print(section, config[section])

    random.seed(42)

    # if config['GENERAL'].get('analysis'):
    # 	analyze()
    # 	return
    # if config['GENERAL'].get('default_test'):
    # 	print("Testing baseline controllers")
    # 	_, _, _, _, world = evaluate()
    # 	# render_world(world=world, name = "test", log_path = "logs")
    # 	return

    # create directory for experiment results
    experiment_dir = Path(
        config["GENERAL"]["logpath"],
        config["GENERAL"]["experiment_name"],
        strftime("%Y-%m-%d-%H-%M", localtime()),
    )
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # copy config file to experiment directory
    with open(args.config) as original_config:
        with Path(experiment_dir, "config.cfg").open("w") as copy_config:
            [copy_config.write(line) for line in original_config]

    experiment_logs = []
    experiment_champions = []

    # champions = {"predators": GeneticProgrammingPopulation(**config['predators']), "prey": GeneticProgrammingPopulation(**config['prey'])}
    if config.get("MAELSTROM"):
        maelstrom = True
        evolver_class = config["GENERAL"].get("evolver_class", Maelstrom)
        config_keyword = "MAELSTROM"
    else:
        maelstrom = False
        evolver_class = config["GENERAL"].get("evolver_class", GeneticProgrammingIsland)
        config_keyword = "ISLAND"

    if config["GENERAL"].get("parallelize_runs"):
        parallel_runs = min(config["GENERAL"]["runs"], multiprocessing.cpu_count())

        if parallel_runs <= (
            multiprocessing.cpu_count() * 0.25
        ):  # allow for maximum of 125% overprovisioning of worker cores
            cores = 1 + (multiprocessing.cpu_count() // parallel_runs)
        else:
            cores = min(1, (multiprocessing.cpu_count() // parallel_runs))

        evolvers = []
        for run in trange(config["GENERAL"]["runs"], unit=" init", position=0):
            evolvers.append(
                evolver_class(
                    **config[config_keyword], **config, cores=cores, position=run + 1
                )
            )
        run_func = evolver_class.run

        with concurrent.futures.ProcessPoolExecutor(parallel_runs) as run_pool:
            # print(runFunc, evolvers)
            evolution_runs = list(run_pool.map(run_func, evolvers))
        experiment_logs = [run.log for run in evolution_runs]
        experiment_champions = []
        for run in evolution_runs:
            run_champions = {}
            for species, population in run.champions.items():
                run_champions[species] = [
                    gene.to_dict() for _, gene in population.items()
                ]
            experiment_champions.append(run_champions)
        del evolvers
        del evolution_runs

    else:
        for run in trange(config["GENERAL"]["runs"], unit=" run"):
            evolver = evolver_class(**config[config_keyword], **config)
            evolver.run()
            experiment_logs.append(evolver.log)
            run_champions = {}
            for species, population in evolver.champions.items():
                run_champions[species] = [
                    gene.to_dict() for key, gene in population.items()
                ]
            experiment_champions.append(run_champions)

    with gzip.open(Path(experiment_dir, "evolutionLog.json.gz"), mode="wt") as file:
        json.dump(experiment_logs, file, separators=(",", ":"))

    competitorsDir = Path(experiment_dir, "competitors")
    competitorsDir.mkdir(parents=True, exist_ok=True)
    for run, competitors in enumerate(experiment_champions):
        with gzip.open(Path(competitorsDir, f"run{run}.json.gz"), mode="wt") as file:
            json.dump(competitors, file, separators=(",", ":"))

    if config["GENERAL"].get("find_local_champions"):
        print("Beginning champion tournament")
        localTournaments(
            experiment_dir, config["GENERAL"]["final_champions"], more_cores=False
        )


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()
