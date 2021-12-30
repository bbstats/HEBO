import os
import sys
import time
from pathlib import Path

import numpy as np

__file__ = "/home/kamild/projects/antigenbinding/random_search/main.py"  # TODO delete. Used for debugging
ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent.parent)
sys.path.insert(0, ROOT_PROJECT)

from task.tools import Absolut
from random_search.old.random_search import RandomSearch
from utilities.config_utils import load_config

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # TODO add ArgumentParser

    config = load_config('/home/rladmin/antigenbinding/random_search/config.yaml')

    absolut_config = {"antigen": None,
                      "path": config['absolut_config']['path'],
                      "process": config['absolut_config']['process'],
                      'startTask': config['absolut_config']['startTask']}

    # Create a directory to save all results
    # try:
    #     os.mkdir(config['save_dir'])
    # except:
    #     raise Exception("Save directory already exists. Choose another directory name to avoid overwriting data")

    antigens_file = '/home/rladmin/antigenbinding/dataloader/antigens_all.txt'
    with open(antigens_file) as file:
        antigens = file.readlines()
        antigens = [antigen.rstrip() for antigen in antigens]
    print(f'Iterating Over All Antigens In File {antigens_file} \n {antigens}')

    for antigen in antigens:
        absolut_config['antigen'] = antigen

        # Defining the fitness function
        absolut_binding_energy = Absolut(absolut_config)


        def function(x):
            x = x.astype(int)
            return absolut_binding_energy.Energy(x)


        binding_energy = []
        num_function_evals = []

        print(f"\nAntigen: {antigen}")

        for i, seed in enumerate(config['random_seeds']):
            start = time.time()
            print(f"\nRandom seed {i + 1}/{len(config['random_seeds'])}")

            np.random.seed(seed)

            _save_dir = os.path.join(
                config['save_dir'],
                f"antigen_{antigen}_kernel_RS_seed_{seed}_cdr_constraint_True_seqlen_{config['sequence_length']}")

            rs = RandomSearch(function=function, dimension=config['sequence_length'], num_iter=config['rs_num_iter'],
                              batch_size=config['rs_batch_size'], save_dir=_save_dir, convergence_curve=True)

            results = rs.run()

            binding_energy.append(results['BestValue'].to_numpy())
            num_function_evals.append(results['Index'].to_numpy())
            print("\nTime taken to run Genetic algorithm for a single seed: {:.0f}s".format(time.time() - start))

        binding_energy = np.array(binding_energy)
        num_function_evals = np.array(num_function_evals)

        np.save(os.path.join(config['save_dir'], "binding_energy_{}.npy".format(antigen)), binding_energy)
        np.save(os.path.join(config['save_dir'], "num_function_evals_{}.npy".format(antigen)), num_function_evals)

        n_std = 1

        plt.figure()
        plt.title(f"Random Search {antigen}")
        plt.grid()
        plt.plot(num_function_evals[0], np.mean(binding_energy, axis=0), color="b")
        plt.fill_between(num_function_evals[0],
                         np.mean(binding_energy, axis=0) - n_std * np.std(binding_energy, axis=0),
                         np.mean(binding_energy, axis=0) + n_std * np.std(binding_energy, axis=0),
                         alpha=0.2, color="b")
        plt.xlabel('Number of function evaluations')
        plt.ylabel('Minimum Binding Energy')
        plt.savefig(os.path.join(config['save_dir'], "binding_energy_vs_funct_evals_{}.png".format(antigen)))
        plt.close()
