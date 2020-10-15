Implementation (pytorch 1.0) of "Do sequence-to-sequence VAEs encode global features of sentences?", [EMNLP2020](https://2020.emnlp.org/papers/main)

Thanks to Junxian He, Bohan Li, and colleagues for the code on which this is based, [available here](https://github.com/bohanli/vae-pretraining-encoder).

# Install

```
conda env create -f environment.yml
conda activate fwp 
```

Install [fasttext](https://github.com/facebookresearch/fastText) in the conda environment, both the standalone C++ tool and the python bindings. For the standalone tool, use `cmake` **but** replace `cmake ..` with `cmake -DCMAKE_INSTALL_PREFIX:PATH="$CONDA_PREFIX" ..`. 

`$FWP_ROOT_EXP` is a global variable holding the root directory of all the experiments. Set it with something like:

```
export FWP_ROOT_EXP="/home/me/vae_experiments" 
```

and [automate](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#saving-environment-variables) the loading of the variable as soon as the environment is activated.

Download the dataset archive [here](https://www.dropbox.com/s/d02e786wulc93oz/datasets.tar.gz?dl=0) and extract it in the `datasets` directory.

# Scripts

The repo contains:

- `config/`: one basic config per dataset, corresponding to the vanilla Bowman & al. (2016) seq2seq VAE.
- `wrap_*.sh arg1 arg2 ...`: train a specific model with specific hyperparameters `arg1 arg2 ...`. 
- `run_*.sh`: run a grid search for a class of model. Call `wrap_*.sh` scripts with the right arguments.
- `scripts/`: various evaluation scripts
- `modules/`: definition of models + tools for scripts

## Train models

Here is a list of models with the scripts to train them. Please look at each script before running: each script does not train all the models as there are too many, so you have to modify most of the scripts. I have put `continue` in the loops so that they are not run before they are modified. You can:

- specify `MODEL` (see below) and `DATASET` (among `yelp`, `amazon`, `yahoo`, `agnews`)
- modify hyperparams search,
- wrap the scripts in SLURM,

In scripts, the variable `MODEL` specifies encoder, pooling function, and decoder at once. Here is the correspondence between the code and the paper:

- `lstm`: `LSTM-last-LSTM` 
- `lstm-max`: `LSTM-max-LSTM`
- `lstm-avg`: `LSTM-avg-LSTM`
- `bow-max`: `BoW-max-LSTM`
- `lstm-uni`: `LSTM-last-Uni` 
- `lstm-max-uni`: `LSTM-max-Uni` 
- `bow-max-uni`: `BoW-max-Uni` 

Since PreAE, PreLM and PreUni methods require a pretraining phase, the corresponding scripts have 2 grid-search loops. Comment one of the two sections or place `continue` in the for loops in order to run the different phases. 

* VAE (various architectures, no pre-training): `run_from_scratch.sh`
* Deterministic AE and PreAE: `run_preae.sh` calls
	1. `wrap_vanilla_ae.sh`: deterministic AE, also pretraining phase of PreAE
	2. `wrap_from_pre.sh`: continue training after resetting the decoder's weights (for PreAE)
* LSTM-LM and PreLM: `run_prelm.sh` calls
	1. `wrap_lstm_lm.sh`: LSTM-LM baseline, also pretraining phase of PreLM
	2. `wrap_from_pre_enc.sh`: same as `wrap_from_pre.sh`, but freeze encoder's weight (for PreLM) *except* linear transformations to get μ and σ, which are trained.
* PreUni: `run_preuni.sh` calls
	1. `wrap_fs.sh`: VAE with a unigram decoder
	2. `wrap_from_pre_uni.sh`: same as `wrap_from_pre_enc.sh`, but the whole encoder is frozen (including linear transformations).

## Evaluations 

I take the example of the `yahoo` dataset throughout. Tables are produced with the following steps:

### Visualize reconstruction (Fig. 1, 2, 3, 4)

1. Compute and store losses per positions: `./scripts/compute_loss_per_pos.sh yahoo`
2. Visualize and save figure: `./scripts/gen_loss_figure.sh yahoo` 

### SSL evaluation (Table 1, 8)

1. Run classifiers or gather stats
	* semi-supervised learning (SSL) eval (Sec. 5): run classifiers on the desired models with `scripts/eval_classification.sh`.
2. Aggregate results in an archive for later analysiS
	* `scripts/analyze.sh yahoo pool_cv`: pool results from all experiment subdirectories that contains `yahoo` into a `.npy` file stored in `npy_archives`. For cross-validation runs.
	* `scripts/analyze.sh yahoo pool_all`: same, but for pooling whole training set runs.
3. Generating latex tables:
	* Generate Table 1 or 8: `scripts/analyze.sh yahoo eval_ssl` (see comments for generating 8)

### Generation evaluation (Table 2)

1. `./scripts/train_fT.sh`: train fastText classifiers (to compute agreement)
2. Modify `./scripts/decode.sh` to run on desired subdirectories, and run, in order to:
	1. Reconstructs (by sampling) documents
	2. Compute agreement using fastText classifiers
	3. Compute exact reconstruction rate
3. Follow the steps to do SSL evaluation: due to poor factoring of code, the same script aggregates all the SSL and decoding results in a single dataframe. Maybe you can avoid doing SSL evaluation by directly running `scripts/analyze.sh yahoo pool_all`, but I have not tried.
4. Estimate NLL and PPL of VAEs: `scripts/eval_iwelbo.sh`
5. Generate Table 2: `scripts/analyze.sh yahoo eval_dec`

# TODO

I will release appendix-related scripts later, but please tell me if you need them fast:

- Add missing scripts for Appendix:
    * Table 3 & 4: KL annealing and original freebits are bad, resetting the decoder matters a lot.
    * Table 5: Datasets characteristics
    * Table 6: Decoding with a VAE that encodes perfectly and only the label
    * Table 7: KL collapse does not prevent mu from being super informative
    * Table 9: All words vs first 3 words
	* Table 10, 11, 12, 13: Cherry picking
- Add preprocessing scripts:
    * Data preprocessing (meanwhile, preprocessed files can be downloaded)
	* `scripts/prepare_vocab.sh`: generate the `vocab.txt` file necessary for decoding (are provided already in the archive)
- Add bibtex reference once it's available
