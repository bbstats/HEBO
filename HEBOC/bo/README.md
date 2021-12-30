First install huggingface (https://huggingface.co/docs/transformers/installation) and clone Protein BERT from (https://huggingface.co/Rostlab/prot_bert).


Also available (/nfs/aiml/asif/ProtBERT)


The BERT encoder transforms a sequence of length L to a representation of size Lx1024. 
To reduce the dimensionality we take CDRH3 data from Absolut map them to representation space and compute 100 principal components.
Later, in a BO loop we reduce the dimensionality by projecting the representation on 100 components.

Finetuning BERT model. 

```
python ./model/huggingface_transformers.py
```

The configuration of BERT is defined in the file as

```
    config = {'path': "/nfs/aiml/asif/ProtBERT",
              'modelname': 'prot_bert_bfd',
              'epochs': 10,
              'batch_size':320,
              'data': {'path': "/nfs/aiml/asif/CDRdata",
                        'modelname': None, 'test_size': 0.2,
                       'antibody': 'Murine', 'antigens': None,
                       'seed':42, 'return_energy': False,
                },
              'mode': 'train',
              'seed' : 42,
              'warmup': 1000,
              'weight_decay': 0.01,
              'logsteps': 200,

        }
```

To switch to a different BERT model change the model name. After finetuning results will be saved in a separate folder in path provided in config.



Computes PCA of BERT representations. This is an optional step just to reduce the dimensionality of feature space.

```
python ./bo/utils.py 
```

For PCA the config is in utils.py

```
    bert_config = { 'datapath': '/nfs/aiml/asif/CDRdata',
                    'path': '/nfs/aiml/asif/ProtBERT',
                   'modelname': 'prot_bert_bfd',
                    'use_cuda': True,
                    'batch_size': 256,
                    'device_ids': [2, 3]
                   }
```

Note: PCA computation can be refactored into a separate file. @Ali once you push repo on HEBO I will do that.



Combinatorial BO using kernels on BERT representation

```
python ./bo/main.py --kernel rbfBERT --modelname prot_bert_bfd

python ./bo/main.py --kernel rbf-pca-BERT --modelname prot_bert_bfd

python ./bo/main.py --kernel rbf-BERT --modelname OutputFinetuneBERTprot_bert_bfd

python ./bo/main.py --kernel rbf-pca-BERT --modelname OutputFinetuneBERTprot_bert_bfd
```

Results will be saved in /home/asif/workspace/antigenbinding/results/BObert/