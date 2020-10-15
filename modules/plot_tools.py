import os
import random
from collections import defaultdict
import numpy as np
import torch
from modules import VAE


def loss_per_ts(model, batch, n_doc=2000, doc_size=-1):
    sent_count = 0
    losses_per_pos = defaultdict(list)
    losses_per_reverse_pos = defaultdict(list)
    sent_lengths = []
    for i in np.random.permutation(len(batch)):
        batch_data = batch[i]
        batch_size, sent_len = batch_data.size()
        if sent_count + batch_size > n_doc:
            break
        sent_len = sent_len-1
        if doc_size > 0 and sent_len != doc_size:
            continue
        sent_count += sent_len
        print(sent_len)
        sent_lengths += [sent_len,]*batch_size
        if isinstance(model, VAE):
            _, loss_rc, _ = model.loss(batch_data, 1.0, sum_over_len=False)
        else:
            loss_rc = model.reconstruct_error(batch_data, sum_over_len=False)
        for pos, v in enumerate(loss_rc.t()):
            losses_per_pos[pos] += v.cpu().numpy().tolist()
        for pos, v in enumerate(loss_rc.t().flip(0)):
            losses_per_reverse_pos[pos] += v.cpu().numpy().tolist()
    first_losses_stats = []
    if doc_size > 0:
        N = doc_size
    else:
        N = 10
    for i in range(N):
        l = np.asarray(losses_per_pos[i])
        # the std within models is ignored in the analysis
        # but might still be useful later.
        first_losses_stats.append((l.mean(), l.std()))
        print(i, first_losses_stats[-1])
    last_losses_stats = []
    for i in range(N):
        l = np.asarray(losses_per_reverse_pos[i])
        last_losses_stats.append((l.mean(), l.std()))
        print(i, last_losses_stats[-1])
    sent_lengths = np.asarray(sent_lengths)
    sent_lengths_stats = (sent_lengths.mean(), sent_lengths.std())

    logs = {
        'first_losses_stats': first_losses_stats,
        'last_losses_stats': last_losses_stats,
        'sent_lengths': sent_lengths_stats,
    }
    return logs



def export_avg_loss_per_ts(model, train_data, device, batch_size,
                           load_path, doc_sizes):
    model.load_state_dict(torch.load(load_path))
    model.eval()
    with torch.no_grad():
        data_batch = train_data.create_data_batch(
            batch_size=batch_size,
            device=device,
            batch_first=True,
        )
        model_dir = os.path.dirname(load_path)
        archive_npy = os.path.join(model_dir, 'val_avg_loss_per_ts.npy')
        random.shuffle(data_batch)
        logs = loss_per_ts(model, data_batch)
        for doc_size in [int(s) for s in doc_sizes.split(',')]:
            logs_ = loss_per_ts(
                model, data_batch, doc_size=doc_size,
            )
            logs['first_losses_stats_' + str(doc_size)] = logs_['first_losses_stats']
            logs['last_losses_stats_' + str(doc_size)] = logs_['last_losses_stats']
        logs['exp_dir'] = model_dir
        np.save(archive_npy, logs)
