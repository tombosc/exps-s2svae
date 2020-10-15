import glob
import re
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

prefix_first = 'first_losses_stats'
prefix_last = 'last_losses_stats'


def extract_stats(archive_fn):
    """ Load archive generated with --export-avg-loss-per-ts.
    """
    archive = np.load(archive_fn)[()]
    # filter in loss information (there might be other stats in the archive)
    filtered = dict()
    for k, v in archive.items():
        if (k.startswith(prefix_first) or k.startswith(prefix_last)):
            filtered[k] = v
    return filtered


def aggreg_stats(stats):
    def min_max_bars(stat_list, key):
        data = []
        for a in stat_list:
            # data is structured as a list of tuples (mean, std) where std is
            # taken over documents, using ONE model.
            # here, we ignore this std, we want std model-wise, after average
            data.append(np.asarray(a[key])[:, 0])
        data = np.asarray(data)
        data_min = data.min(0).reshape((1, -1))
        data_max = data.max(0).reshape((1, -1))
        data_mean = data.mean(0)
        data_bars = np.abs(data_mean - np.concatenate([data_min, data_max], 0))
        return data_mean, data_bars, data
    keys = stats[0].keys()
    return {k: min_max_bars(stats, k) for k in keys}


re_seed = re.compile('_s(ee)?d[0-9]+')


def get_results(root_dir):
    """ Browse directories, extract stats and aggregate by seeds.
    """
    print("Browsing ", root_dir)
    all_fn = glob.glob(root_dir + '/*/*.npy')
    list_stats = defaultdict(list)
    aggregated_stats = []
    # aggregate the stats by seeds
    for fn in all_fn:
        try:
            stats = extract_stats(fn)
        except:
            print("Failed", fn)
            continue
        seedless_fn = re.sub(re_seed, '', fn)
        list_stats[seedless_fn].append(stats)
    for k, v in list_stats.items():
        new_fn = Path(k).parts[-2]  # -1 is the filename
        aggregated_stats.append((new_fn, aggreg_stats(v)))
    return aggregated_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dirs", type=str, nargs='+',
                        help="List of absolute paths of exps")
    parser.add_argument("fig_out_fn", type=str)
    args = parser.parse_args()
    data_key = prefix_first + '_16'
    aggregated_stats = []
    for subdir in args.exp_dirs:
        aggregated_stats += get_results(subdir)
    # list all the exp_dirs to pool (without the seed part in the filename),
    # starting with the baseline
    # important for computing relative improvement
    labels_to_fn = {
        'LSTM': 'lm_baseline',
        'AE': 'ae_lstm_el-1_nz16',
        'VAE last-PreAE ($\lambda=2$)': 'from_pre_fb1_2_warmup0_lstm_el-1_nz16',
        'VAE last-PreAE ($\lambda=8$)': 'from_pre_fb1_8_warmup0_lstm_el-1_nz16',
        'VAE max ($\lambda=2$)': 'fs_fb1_2_wu0_lstm_max_el-1_nz16',
        'VAE max ($\lambda=8$)': 'fs_fb1_8_wu0_lstm_max_el-1_nz16',
    }

    raw_aggregated_stats = aggregated_stats
    aggregated_stats = []
    labels = []
    for i, (label, lookup) in enumerate(labels_to_fn.items()):
        found = False
        for name, stats in raw_aggregated_stats:
            if lookup == name:
                print("Found", lookup, label, name)
                labels.append(label)
                aggregated_stats.append(stats)
                found = True
        if not found:
            # it is crucial to find the first file because it is the baseline
            # for the RHS plot => exception
            # in other cases, it is not so important
            if i == 0:
                raise ValueError("Baseline ", label, "not found")
            else:
                print(label, " not found")
    print(labels)
    rc('text', usetex=True)
    baseline_stats = aggregated_stats[0]
    range_ = list(range(len(aggregated_stats[0][data_key][0])))
    figure, (ax1, ax2) = plt.subplots(
        1, 2, sharey=False, figsize=(12, 4),
    )
    ax1.set_ylabel('reconstruction loss')
    ax1.set_xlabel('position')
    for f, label in zip(aggregated_stats, labels):
        ax1.errorbar(range_, f[data_key][0], yerr=f[data_key][1], label=label)

    ax2.set_ylabel('relative improvement')
    ax2.set_xlabel('position')
    for f, label in zip(aggregated_stats, labels):
        y = (baseline_stats[data_key][0] - f[data_key][0])
        y = (y * (y > 0)) / baseline_stats[data_key][0]
        ax2.plot(range_, y, label=label)
    ax2.legend()
    plt.show()
    figure.savefig(args.fig_out_fn, bbox_inches='tight')
