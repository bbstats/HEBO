import os
import sys
import time

import numpy as np

from pathlib import Path

import matplotlib
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# __file__ = "/home/kamild/projects/antigenbinding/genetic_algorithm/main.py"  # TODO delete. Used for debugging
# ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent.parent)
# sys.path.insert(0, ROOT_PROJECT)

from task.tools import Absolut
from utilities.config_utils import load_config
from genetic_algorithm.actor import GeneticAlgorithmActor
from environment.binding_environment import BindingEnvironment
import pandas as pd
from utilities.config_utils import save_config
from utilities.aa_utils import aa_to_idx, indices_to_aa_seq
from utilities.results_logger import ResultsLogger


def evaluate_batch(X, fitness, res, time_to_generate_population=0, num_funct_evals=0, best_function=None, best_sequence=None):
    #############################################################
    # Function to evaluate a batch of samples
    start = time.time()
    temp = X.copy()
    batch_size = len(X)
    time_to_evaluate_population = time.time() - start

    mean_time_per_sample = (time_to_generate_population + time_to_evaluate_population) / batch_size

    for X_idx, res_idx in enumerate(range(num_funct_evals, num_funct_evals + batch_size)):
        aa_seq = indices_to_aa_seq(temp[X_idx])
        binding_energy = fitness[X_idx]

        # Initialisation during first function evaluation
        if (best_function is None and best_sequence is None):
            best_function = binding_energy
            best_sequence = aa_seq

        # Check if binding energy of current sequence is lower than the binding energy of the best sequence
        elif (binding_energy < best_function):
            best_function = binding_energy
            best_sequence = aa_seq

        # Append results 'Index', 'LastValue', 'BestValue', 'Time', 'LastProtein', 'BestProtein'

        res.iloc[res_idx + 1] = {'Index': int(res_idx) + 1, 'LastValue': binding_energy,
                                      'BestValue': best_function, 'Time': mean_time_per_sample,
                                      'LastProtein': aa_seq, 'BestProtein': best_sequence, }

    return res, best_function, best_sequence



def summarisation(res, save_dir, num_funct_evals, convergence_curve=True):

    if not os.path.exists(save_dir):
      os.mkdir(save_dir)
    res.to_csv(os.path.join(save_dir, 'results.csv'))

    if convergence_curve == True:
        plt.figure()
        plt.title("Genetic Algorithm Binding energy curve")
        plt.grid()
        plt.plot(res.iloc[1:num_funct_evals + 1]['Index'].astype(int),
                 res.iloc[1:num_funct_evals + 1]['BestValue'])
        plt.xlabel('Number of function evaluations')
        plt.ylabel('Minimum Binding Energy')
        plt.savefig(os.path.join(save_dir, "binding_energy_vs_funct_evals.png"))
        plt.close()

    return

def progress(count, total, status=''):
    bar_len = 50
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '|' * filled_len + '_' * (bar_len - filled_len)

    sys.stdout.write('\r%s %s%s %s' % (bar, percents, '%', status))
    sys.stdout.flush()


def calculate_total_num_funct_evals(param):
    population_size = param['batch_size']
    num_iterations = param['max_num_iterations']
    num_elite = int(population_size * param['experiment_config']['elite_ratio'])
    if num_elite % 2 != 0:  # Ensure that the number of elite samples is even
        num_elite += 1

    return population_size + num_iterations * (population_size - num_elite)

if __name__ == '__main__':
    # TODO add ArgumentParser

    config = load_config('config_ga.yaml')

    # Main Parameters

    absolut_config = {"antigen": None,
                      "path": config['absolut_config']['path'],
                      "process": config['absolut_config']['process'],
                      'startTask': config['absolut_config']['startTask']}

    # GA parameters
    algorithm_parameters = {'population_size': config['batch_size'], \
                            'mutation_probability': 1 / config['sequence_length'], \
                            'elite_ratio': config['experiment_config']['elite_ratio'], \
                            'crossover_probability': config['experiment_config']['crossover_probability'], \
                            'parents_portion': config['experiment_config']['parents_portion'], \
                            'crossover_type': config['experiment_config']['crossover_type'],
                            'sequence_length': config['sequence_length']}

    # Print the number of black-box function evaluations per antigen per random seed
    print(
        f"\nNumber of function evaluations per random seed: {calculate_total_num_funct_evals(config)}")

    # Create a directory to save all results
    """
    try:
        os.mkdir(config['save_dir'])
    except:
        raise Exception("Save directory already exists. Choose another directory name to avoid overwriting data")
    """

    with open('dataloader/all_antigens.txt') as file:
        antigens = file.readlines()
        antigens = [antigen.rstrip() for antigen in antigens]
    print(f"Running over all input antigens from file: \n \n {antigens} \n")
    for antigen in tqdm(antigens):
        absolut_config['antigen'] = antigen

        # Defining the fitness function
        absolut_binding_energy = Absolut(absolut_config)

        def function(x):
            x = x.astype(int)
            return absolut_binding_energy.Energy(x)

        environment_config = {'n_sequences': config['batch_size'],
                              'dimensions': config['sequence_length'],
                              'model_tag': antigen,
                              'environment_name': config['environment_name']}

        env = BindingEnvironment(environment_config, function)

        binding_energy = []
        num_function_evals = []

        print(f"\nAntigen: {antigen}")

        for i, seed in enumerate(config['random_seeds']):
            start = time.time()
            print(f"\nRandom seed {i + 1}/{len(config['random_seeds'])}")

            np.random.seed(seed)
            _save_dir = os.path.join(
                config['save_dir'],
                f"antigen_{antigen}_kernel_GA_seed_{seed}_cdr_constraint_True_seqlen_{config['sequence_length']}")


            if os.path.exists(_save_dir):
                print(f"antigen_{antigen}_kernel_GA_seed_{seed}_cdr_constraint_True_seqlen_{config['sequence_length']}")
                continue

            actor = GeneticAlgorithmActor(algorithm_parameters)
            max_num_funct_evals = calculate_total_num_funct_evals(config)

            # results = ResultsLogger(max_num_funct_evals)  or  initialise instance outside of loop and just call
            # results.reset() here

            res = pd.DataFrame(np.nan, index=np.arange(int(max_num_funct_evals) + 1),
                                    columns=['Index', 'LastValue',
                                             'BestValue', 'Time',
                                             'LastProtein',
                                             'BestProtein'])

            dim = 11

            max_num_iter = int(config['max_num_iterations'])

            #############################################################
            # Variable to keep track of total number of function evaluations
            num_funct_evals = 0
            best_sequence = None
            best_function = None

            #############################################################
            # Initial Population. Last column stores the fitness
            start_generation = time.time()
            population_to_eval = actor.suggest()
            _, fitness, _, _ = env.step(population_to_eval)
            end_generation = time.time()
            res, best_function, best_sequence = evaluate_batch(population_to_eval, fitness, res, end_generation-start_generation, num_funct_evals, best_function, best_sequence)
            actor.observe(fitness)
            # results.append_batch()
            num_funct_evals += len(population_to_eval)

            ##############################################################
            gen_num = 1

            while gen_num <= max_num_iter:

                progress(gen_num, max_num_iter, status="GA is running...")

                summarisation(res, _save_dir, num_funct_evals)
                
                start_generation = time.time()
                population_to_eval = actor.suggest()
                _, fitness, _, _ = env.step(population_to_eval)
                end_generation = time.time()
                res, best_function, best_sequence = evaluate_batch(population_to_eval, fitness, res, end_generation-start_generation, num_funct_evals, best_function, best_sequence)
                actor.observe(fitness)
                # results.append_batch()
                num_funct_evals += len(population_to_eval)

                gen_num += 1

            sys.stdout.write('\r The best solution found:\n %s' % (res.iloc[-1]['BestProtein']))
            sys.stdout.write('\n\n Objective function:\n %s\n' % (res.iloc[-1]['BestValue']))
            sys.stdout.flush()

            #results.save(_save_dir)
            summarisation(res, _save_dir, num_funct_evals)
            save_config(algorithm_parameters, os.path.join(_save_dir, 'config.yaml'))
            #np.save(os.path.join(_save_dir, 'final_population.npy'), population)

            binding_energy.append(res['BestValue'].to_numpy())
            num_function_evals.append(res['Index'].to_numpy())
            print("\nTime taken to run Genetic algorithm for a single seed: {:.0f}s".format(time.time() - start))

        binding_energy = np.array(binding_energy)
        num_function_evals = np.array(num_function_evals)

        np.save(os.path.join(config['save_dir'], f"binding_energy_{antigen}.npy"), binding_energy)
        np.save(os.path.join(config['save_dir'], f"num_function_evals_{antigen}.npy"), num_function_evals)

        n_std = 1

        plt.figure()
        plt.title(f"Genetic Algorithm {antigen}")
        plt.grid()
        plt.plot(num_function_evals[0], np.mean(binding_energy, axis=0), color="b")
        plt.fill_between(num_function_evals[0],
                         np.mean(binding_energy, axis=0) - n_std * np.std(binding_energy, axis=0),
                         np.mean(binding_energy, axis=0) + n_std * np.std(binding_energy, axis=0),
                         alpha=0.2, color="b")
        plt.xlabel('Number of function evaluations')
        plt.ylabel('Minimum Binding Energy')
        plt.savefig(os.path.join(config['save_dir'], f"binding_energy_vs_funct_evals_{antigen}.png"))
        plt.close()
