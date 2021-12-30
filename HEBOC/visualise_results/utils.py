import os
import glob

import pandas as pd


def load_results(results_dir):
    # Create a Pandas dataframe to store all results
    columns = ['Method', 'Antigen', 'Seed', 'Num BB Evals', 'Suggest Time', 'Last Protein', 'Last Binding Energy',
               'Last Charge', 'Last Hydropathicity', 'Last Instability Index', 'Best Protein', 'Best Binding Energy',
               'Best Charge', 'Best Hydropathicity', 'Best Instability Index']
    results = pd.DataFrame(columns=columns)

    # Get the name of every folder in the results directory
    folders = glob.glob(os.path.join(results_dir, '*'))

    for folder in folders:
        method = folder.split('/')[-1]
        subfolders = glob.glob(os.path.join(folder, '*'))

        for subfolder in subfolders:
            _name = subfolder.split('/')[-1]
            antigen = _name.split('antigen_')[1].split('_kernel')[0]
            seed = int(_name.split('_seed_')[1].split('_cdr_')[0])
            # method = _name.split('_kernel_')[1].split('_seed_')[0]
            df = pd.read_csv(os.path.join(subfolder, 'results.csv'))

            # Add method, antigen and seed column
            df['Method'] = len(df['Num BB Evals']) * [method]
            df['Antigen'] = len(df['Num BB Evals']) * [antigen]
            df['Seed'] = len(df['Num BB Evals']) * [seed]

            df = df[columns]

            # results.append(df, ignore_index=True)
            results = pd.concat([results, df], ignore_index=True, sort=False)

    return results

import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    results_dir = '/home/rladmin/antigenbinding/new_results_test_vis/'

    results = load_results(results_dir)

    # Create a visualization
    # sns.relplot(
    #     data=results, kind="line",
    #     x="time", y="firing_rate", col="align",
    #     hue="choice", size="coherence", style="choice",
    #     facet_kws=dict(sharex=False),
    # )
    sns.relplot(
        data=results, kind="line",
        x='Num BB Evals', y='Best Binding Energy', hue = "Method", col="Antigen",col_wrap=4, # col="align",
        #hue="choice", size="coherence", style="choice",
        facet_kws=dict(sharex=False),
    )
    plt.show()
# Implementation for old format of results
# def load_results(results_dir):
#     # Create a Pandas dataframe to store all results # TODO add developability scores
#     columns = ['Method', 'Antigen', 'Seed', 'Num BB Evals', 'Proposed Sequence', 'Binding Energy', 'Best Sequence',
#                'Best Binding Energy', 'Suggest time']
#     results = pd.DataFrame(columns=columns)
#
#     # Get the name of every folder in the results directory
#     folders = glob.glob(os.path.join(results_dir, '*'))
#
#     for folder in folders:
#         # method_name = folder.split('/')[-1]
#         subfolders = glob.glob(os.path.join(folder, '*'))
#
#         for subfolder in subfolders:
#             _name = subfolder.split('/')[-1]
#             antigen = _name.split('antigen_')[1].split('_kernel')[0]
#             seed = int(_name.split('_seed_')[1].split('_cdr_')[0])
#             method = _name.split('_kernel_')[1].split('_seed_')[0]
#             df = pd.read_csv(os.path.join(subfolder, 'results.csv'))
#
#             df = df.rename(columns={'Index': 'Num BB Evals',
#                                     'LastValue': 'Binding Energy',
#                                     'BestValue': 'Best Binding Energy',
#                                     'LastProtein': 'Proposed Sequence',
#                                     'BestProtein': 'Best Sequence',
#                                     'Time': 'Suggest time'})
#
#             # Remove rows with NaN and reindex
#             df = df.dropna(axis=0)
#             df.reset_index(inplace=True, drop=True)
#
#             # Drop unnamed columns
#             if df.columns.str.contains('^Unnamed').sum() != 0:
#                 df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
#
#             # Add method, antigen and seed column
#             df['Method'] = len(df['Num BB Evals']) * [method]
#             df['Antigen'] = len(df['Num BB Evals']) * [antigen]
#             df['Seed'] = len(df['Num BB Evals']) * [seed]
#
#             df = df[columns]
#
#             # results.append(df, ignore_index=True)
#             results = pd.concat([results, df], ignore_index=True, sort=False)
#
#     return results
