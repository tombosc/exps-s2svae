params={
    'enc_type': 'lstm',
    'dec_type': 'lstm',
    'pred_linear_bias': True,
    'nz': 16,
    'ni': 200,
    'enc_nh': 512,
    'dec_nh': 512,
    'dec_dropout_in': 0.5,
    'dec_dropout_out': 0.5,
    'dec_dropout': 0.5,  # for unigram decoders
    'batch_size': 64,
    'epochs': 200,
    'test_nepoch': 5,
    'train_data': 'datasets/amazon/train.txt.100k',
    'val_data': 'datasets/amazon/valid.txt.10k',
    'test_data': 'datasets/amazon/test.txt.10k',
    'vocab_file': 'datasets/amazon/vocab.txt',
    "label": True,
}
