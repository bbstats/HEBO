from pathlib import Path
import sys
import os
ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent.parent)
sys.path.insert(0, ROOT_PROJECT)

from bo.botask import BOTask as CDRBO
from bo.optimizer import Optimizer
import os
import time, datetime
from bo.random_seed_config import *
import torch
import pandas as pd
import numpy as np
from bo.utils import save_w_pickle, load_w_pickle
import pdb
import argparse

class BOExperiments:
    def __init__(self,
                 config,
                 cdr_constraints,
                 seed):
        '''

        :param config: dictionary of parameters for BO
                acq: choice of the acquisition function
                ard: whether to enable automatic relevance determination
                save_path: path to save model and results
                kernel_type: choice of kernel
                normalise: normalise the target for the GP
                batch_size: batch size for BO
                max_iters: maximum evaluations for BO
                n_init: number of initialising random points
                min_cuda: number of initialisation points to use CUDA
                device: default 'cpu' if GPU specify the id
                seq_len: length of seqence for BO
                bbox: dictionary of parameters of blackbox
                    antigen: antigen to use for BO
                seed: random seed

        '''
        self.config = config
        self.seed = seed
        self.cdr_constraints = cdr_constraints
        # Sanity checks
        assert self.config['acq'] in ['ucb', 'ei', 'thompson'], f"Unknown acquisition function choice {self.config['acq']}"
        if 'search_strategy' in self.config:
            self.search_strategy = self.config['search_strategy']
            assert self.search_strategy in ['glocal', 'local'], print(f"{self.search_strategy} not in ['glocal', 'local']")
        else:
            self.search_strategy = 'local'

        print(f"Search Strategy {self.search_strategy}")

        if self.config['kernel_type'] is None:
            self.config['kernel_type'] = 'transformed_overlap'
            print(f"Kernel Not Specified Using Default {self.config['kernel_type']}")


        self.path = f"{self.config['save_path']}/antigen_{self.config['bbox']['antigen']}" \
                    f"_kernel_{self.config['kernel_type']}_seed_{self.seed}" \
                    f"_cdr_constraint_{self.cdr_constraints}_seqlen_{self.config['seq_len']}"

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.res = pd.DataFrame(np.nan, index=np.arange(int(self.config['max_iters']*self.config['batch_size'])),
                           columns=['Index', 'LastValue', 'BestValue', 'Time', 'LastProtein', 'BestProtein'])

        self.nm_AAs = 20
        self.n_categories = np.array([self.nm_AAs] * self.config['seq_len'])
        self.start_itern = 0
        self.f_obj = CDRBO(self.config['device'], self.n_categories, self.config['seq_len'], self.config['bbox'], False)

    def load(self):
        res_path = os.path.join(self.path, 'results.csv')
        optim_path = os.path.join(self.path, 'optim.pkl')
        if os.path.exists(optim_path):
            optim = load_w_pickle(optim_path)
        else:
            optim = None
        if os.path.exists(res_path):
            self.res = pd.read_csv(res_path, usecols=['Index', 'LastValue', 'BestValue', 'Time', 'LastProtein', 'BestProtein'])
            self.start_itern = len(self.res) - self.res['Index'].isna().sum() + 1
        return optim

    def save(self, optim):
        optim_path = os.path.join(self.path, 'optim.pkl')
        res_path = os.path.join(self.path, 'results.csv')
        save_w_pickle(optim, optim_path)
        self.res.to_csv(res_path)

    def results(self, optim, x, itern, rtime):
        Y = np.array(optim.casmopolitan.fX)
        if Y[:itern].shape[0]:
            argmin = np.argmin(Y[:itern * self.config['batch_size']])
            x_best = ''.join([self.f_obj.fbox.idx_to_AA[j] for j in optim.casmopolitan.X[:itern * self.config['batch_size']][argmin].flatten()])
            # sequential
            if self.config['batch_size'] == 1:
                self.res.iloc[itern, :] = [itern, float(Y[-1]), float(np.min(Y[:itern])), rtime, self.f_obj.idx_to_seq(x)[0], x_best]
            # batch
            else:
                for idx, j in enumerate(range(itern * self.config['batch_size'], (itern + 1) * self.config['batch_size'])):
                    self.res.iloc[j, :] = [j, float(Y[-idx]), float(np.min(Y[:itern * self.config['batch_size']])), rtime,
                                                self.f_obj.idx_to_seq(x)[idx], x_best]

    def run(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.config['kernel_type'] in ['rbfBERT', 'rbf-pca-BERT', 'cosine-BERT', 'cosine-pca-BERT']:
            kwargs = {
                'length_max_discrete': self.config['seq_len'],
                'device': self.config['device'],
                'search_strategy': self.search_strategy,
                'seed': self.seed,
                'BERT_model' : self.config['BERT']['model'],
                'BERT_tokeniser' : self.config['BERT']['tokeniser'],
                'BERT_batchsize' : self.config['BERT']['batch_size'],
                'antigen': self.config['bbox']['antigen'],
            }
        else:
            kwargs = {
                'length_max_discrete': self.config['seq_len'],
                'device': self.config['device'],
                'search_strategy': self.search_strategy,
                'seed': self.seed,
            }

        if self.config['resume']:
            optim = self.load()
        else:
            optim = None

        if not optim:
            optim = Optimizer(self.n_categories, min_cuda=self.config['min_cuda'],
                              n_init=self.config['n_init'], use_ard=self.config['ard'],
                              acq=self.config['acq'],
                              cdr_constraints=self.cdr_constraints,
                              normalise=self.config['normalise'],
                              kernel_type=self.config['kernel_type'],
                              noise_variance=float(self.config['noise_variance']),
                              batch_size = self.config['batch_size'],
                              alphabet_size=self.nm_AAs,
                              **kwargs
                              )

        for itern in range(self.start_itern, self.config['max_iters']):
            start = time.time()
            x_next = optim.suggest(self.config['batch_size'])
            y_next = self.f_obj.compute(x_next)
            self.save(optim)
            optim.observe(x_next, y_next)
            end = time.time()
            self.results(optim, x_next, itern, rtime=end-start)

from bo.utils import get_config

if __name__ == '__main__':

    parser = argparse.ArgumentParser(add_help=True, description='Antigen-CDR3 binding prediction using high dimensional BO')
    parser.add_argument('--antigens_file', type=str, default='/home/asif/workspace/antigenbinding/dataloader/all_antigens.txt', help='List of Antigen to perform BO')
    parser.add_argument('--seed', type=int, default=42, help='initial seed setting')
    parser.add_argument('--n_trails', type=int, default=3, help='number of random trials')
    parser.add_argument('--resume', type=bool, default=False, help='flag to resume training')
    parser.add_argument('--resume_trial', type=int, default=0, help='resume trail for training')
    parser.add_argument('--cdr_constraints', type=bool, default=True, help='constraint local search')
    parser.add_argument('--device_ids', type=list, default=[1,2], help='gpu device ids used for training')
    parser.add_argument('--config', type=str, default='/home/asif/workspace/antigenbinding/bo/config.yaml', help='Configuration File')
    parser.add_argument('--kernel', type=str, default='transformed_overlap', help='GP Kernel')
    parser.add_argument('--modelname', type=str, default='prot_bert_bfd', help='BERT model name')

    args = parser.parse_args()
    config = get_config(args.config)
    config['resume'] = args.resume
    config['kernel_type'] = args.kernel

    # with open(args.antigens_file) as file:
    #      antigens = file.readlines()
    #      antigens = [antigen.rstrip() for antigen in antigens]
    #
    # print(f'Iterating Over All Antigens In File {args.antigens_file} \n {antigens}')
    antigens = ['1ADQ_A', '1FBI_X', '1H0D_C', '1NSN_S', '1OB1_C', '1WEJ_F', '2YPV_A', '3RAJ_A', '3VRL_C', '2DD8_S', '1S78_B', '2JEL_P']
    if config['kernel_type'] in ['rbfBERT','rbf-pca-BERT', 'cosine-BERT', 'cosine-pca-BERT']:
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(str(id) for id in args.device_ids)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        bert_config = {'path': '/nfs/aiml/asif/ProtBERT',
                       'modelname': args.modelname,
                       }
        device = 'cuda:1'
        from transformers import pipeline, \
            AutoTokenizer, \
            Trainer, \
            AutoModel

        BERT = {
            'tokeniser': AutoTokenizer.from_pretrained(f"{bert_config['path']}/{bert_config['modelname']}"),
            'model': AutoModel.from_pretrained(f"{bert_config['path']}/{bert_config['modelname']}").to(device),
            'batch_size': 16,
            'use_pca': False
        }
        config['BERT'] = BERT

    for antigen in antigens:
        start_antigen = time.time()
        seeds = list(range(args.seed, args.seed + args.n_trails))
        t = args.resume_trial
        while(t < args.n_trails):
            print(f"Starting Trial {t+1} for antigen {antigen}")
            config['bbox']['antigen'] = antigen
            boexp = BOExperiments(config, args.cdr_constraints, seeds[t])
            boexp.run()
            del boexp
            torch.cuda.empty_cache()
            end_antigen = time.time()
            print(f"Time taken for antigen {antigen} trial {t} = {end_antigen - start_antigen}")
            t += 1
        args.resume_trial = 0
    print('BO finished')