import os
import glob
import pandas as pd

import sys
from pathlib import Path
ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent.parent)
sys.path.insert(0, ROOT_PROJECT)

from utilities.results_logger import ResultsLogger


def compute_developability_scores_for_existing_results(results_path, save_dir):
    # Load dataframe
    df = pd.read_csv(os.path.join(results_path))

    # Remove rows with NaN and reindex
    df = df.dropna(axis=0)
    df.reset_index(inplace=True, drop=True)

    # Drop unnamed columns
    if df.columns.str.contains('^Unnamed').sum() != 0:
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Iterate through all results and compute developability scores for them

    results = ResultsLogger(len(df))

    for index, row in df.iterrows():
        results._append(protein=row['LastProtein'], binding_energy=row['LastValue'], suggest_time=row['Time'],
                        num_bb_evals=row['Index'])

    results.save(save_dir)


if __name__ == '__main__':
    # TODO make sure you backup all data before running this script
    results_dir = '/home/rladmin/antigenbinding/results_test_vis/'
    save_dir = '/home/rladmin/antigenbinding/new_results_test_vis/'

    try:
        os.mkdir(save_dir)
    except:
        raise Exception('Save dir already exists. Choose another save dir.')

    # Get the name of every folder in the results directory
    folders = glob.glob(os.path.join(results_dir, '*'))

    for folder in folders:
        folder_name = folder.split('/')[-1]
        os.mkdir(os.path.join(save_dir, folder_name))

        subfolders = glob.glob(os.path.join(folder, '*'))

        for subfolder in subfolders:
            subfolder_name = subfolder.split('/')[-1]
            os.mkdir(os.path.join(save_dir, folder_name, subfolder_name))

            _results_path = os.path.join(subfolder, 'results.csv')
            _save_dir = os.path.join(save_dir, folder_name, subfolder_name)
            compute_developability_scores_for_existing_results(_results_path, _save_dir)
