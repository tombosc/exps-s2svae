""" Store hyperparameters.
"""
from collections import Counter

# This list is used for analyzing results with pandas.
# It could be gathered smartly by going through models and optim params.
# For now, it is not automatic. So if you add an hyperparameter, add it here.

hyperparameters = [
    'enc_type', 'dec_type', 'warm_up', 'pooling', 'reset_dec', 'pre', 'fb',
    'encode_length', 'freeze_encoder', 'target_kl', 'nz', 'vanilla_ae',
    'unidec', 'skip_first_word', 'reconstruct_partial', 'seed',
    'reconstruct_partial_from', 'enc_reverse', 'reconstruct_random_words',
    'freeze_encoder_exc', 'enc_dropout', 'dec_dropout', 'opt', 'fix_var',
    'dec_dropout_in', 'dec_dropout_out', 'pred_linear_bias',
    'enc_nh', 'dec_nh', 'lr', 'batch_size', 'kl_start', 'discrete',
]
assert(len(hyperparameters) == len(set(hyperparameters)))

all_hp_except_seed = [e for e in hyperparameters if e != 'seed']


def check_correct_groupby(df, grouping_params, changing_params):
    """ Run a sanity check on the dataframe, to verify that when we
    df.groupby(grouping_params) is run, the hyperparameters are only those in
    changing_params. (i.e. only 'seed' is in changing_params)

    When the sanity check fails, it lists the directories that contain the
    faulty experiments.
    """
    grouped = df.groupby(grouping_params + changing_params)
    grouped_mean = grouped.mean()
    if len(grouped_mean) < len(df):
        print("Error: either you have introduced new hyperparameters"
              "without referencing them; or you are trying to load"
              "identical experiments from different directories.")
        print("Here are the duplicated runs:")
        df_reindexed = df.set_index(grouped_mean.index.names)
        idx_counts = Counter(df_reindexed.index)
        duplicated = [i for i, c in idx_counts.items() if c > 1]
        first_duplicated = df_reindexed.loc[duplicated[0]]
        for subdir in first_duplicated['exp_dir'].values:
            print("- ", subdir)
        exit(1)
    elif len(grouped_mean) > len(df):
        print("Unknown error")  # I guess this should never happen
        exit(1)
