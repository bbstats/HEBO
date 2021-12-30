import os
import numpy as np
import pandas as pd

from task.utils import compute_scores
from utilities.aa_utils import indices_to_aa_seq


class ResultsLogger:

    def __init__(self, size):
        self.size = size
        self._idx = 0
        self.columns = ['Num BB Evals', 'Suggest Time', 'Last Protein', 'Last Binding Energy', 'Last Charge',
                        'Last Hydropathicity', 'Last Instability Index', 'Best Protein', 'Best Binding Energy',
                        'Best Charge', 'Best Hydropathicity', 'Best Instability Index']

        self.res = pd.DataFrame(np.nan, index=np.arange(1, self.size + 1), columns=self.columns)

    def _append(self, protein, binding_energy, suggest_time, num_bb_evals):

        scores = compute_scores([protein])
        charge = scores['charge'][0]
        hydropathicity = scores['hydropathicity'][0]
        instability = scores['instability'][0]

        if self._idx == 0:
            best_binding_energy = binding_energy
            best_protein = protein
            best_charge = charge
            best_hydrophobicity = hydropathicity
            best_instability = instability

        else:
            best_idx = self.res.iloc[:self._idx]['Last Binding Energy'].argmin()
            best_binding_energy = self.res.iloc[best_idx]['Last Binding Energy']
            best_protein = self.res.iloc[best_idx]['Last Protein']
            best_charge = self.res.iloc[best_idx]['Last Charge']
            best_hydrophobicity = self.res.iloc[best_idx]['Last Hydropathicity']
            best_instability = self.res.iloc[best_idx]['Last Instability Index']

            if best_binding_energy > binding_energy:
                best_binding_energy = binding_energy
                best_protein = protein
                best_charge = charge
                best_hydrophobicity = hydropathicity
                best_instability = instability

        self.res.iloc[self._idx] = [int(num_bb_evals), suggest_time, protein, binding_energy, charge, hydropathicity,
                                    instability, best_protein, best_binding_energy, best_charge, best_hydrophobicity,
                                    best_instability]

        self._idx += 1

    def append(self, X, binding_energies, suggest_time, num_bb_eval):

        proteins = [indices_to_aa_seq(x) for x in X]
        suggest_times = [suggest_time / len(X) for _ in X]
        num_bb_evals = [i for i in range(num_bb_eval - len(X) + 1, num_bb_eval + 1)]

        for protein_, binding_energy_, suggest_time_, num_bb_evals_ in zip(proteins, binding_energies, suggest_times,
                                                                           num_bb_evals):
            self._append(protein_, binding_energy_, suggest_time_, num_bb_evals_)

    def save(self, save_dir):
        self.res.to_csv(os.path.join(save_dir, 'results.csv'), index=False)

    def reset(self):
        self._idx = 0
        self.res = pd.DataFrame(np.nan, index=np.arange(1, self.size + 1), columns=self.columns)


if __name__ == '__main__':

    import random
    from utilities.aa_utils import aas

    n = 20

    logger = ResultsLogger(n)

    for i in range(n):
        logger._append(protein=''.join(random.choice(aas) for _ in range(11)), binding_energy=np.random.randn(),
                       suggest_time=np.random.random(), num_bb_evals=i + 1)

    logger.save('/home/rladmin/')