from types import SimpleNamespace
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

fix_seed = 2024
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# ===================== Basic Configuration =====================
# is_training: 1 for training+testing, 0 for testing only
# train_only: Whether to train without validation
# model_id: Experiment identifier
# model: Model name
# des: Experiment description
# loss: Loss function
# lradj: Learning rate adjustment strategy
# use_amp: Whether to use mixed precision
# test_flop: Whether to test FLOPs
# checkpoints: Model save path
# embed: Time feature encoding method
#
# ===================== Data Loading =====================
# data: Dataset name
# root_path: Dataset root directory
# data_path: Data file name
# features: Task type (M/S/MS)
# target: Target column
# freq: Time granularity
#
# ===================== Prediction Task Parameters =====================
# seq_len: Input sequence length
# label_len: Label length
# pred_len: Prediction length
# enc_in: Input feature dimension
#
# ===================== Patching Related =====================
# patch_len: Patch length
# stride: Patch stride
# padding_patch: Patch padding method
#
# ===================== Moving Average Related =====================
# ma_type: Moving average type
# alpha, beta: Smoothing coefficients
#
# ===================== Optimizer and Training =====================
# num_workers: Number of data loading threads
# itr: Number of experiment iterations
# train_epochs: Number of training epochs
# batch_size: Batch size
# patience: Early stopping patience
# learning_rate: Learning rate
#
# ===================== LTSF Model Specific Parameters =====================
# num_heads: Number of xLSTM heads
# layers1: Number of sLSTM layers
# layers2: Number of mLSTM layers
# hidden_size: xLSTM hidden size
# fc1_size: First FC layer size in KAN
# fc2_size: KAN output layer size
# revin: Whether to use RevIN
# revin_affine: RevIN affine transformation
# revin_subtract_last: RevIN subtract last value
#
# ===================== GPU Related Parameters =====================
# use_gpu: Whether to use GPU
# gpu: GPU device ID
# use_multi_gpu: Whether to use multiple GPUs
# devices: Multiple GPU device IDs

args = SimpleNamespace(
    # ===== Basic Configuration =====
    is_training=1,
    train_only=False,
    model_id='DIR_Net',
    model='DIR_Net',
    des='test',
    loss='mse',
    lradj='type1',
    use_amp=False,
    test_flop=False,
    checkpoints='autodl-tmp/emLSTM-TCN/checkpoints/',
    embed='timeF',

    # ===== Data Loading =====
    data='Solar',
    root_path='autodl-tmp/emLSTM-TCN/dataset',
    data_path='solar.txt',
    features='M',
    target='OT',
    freq='h',
    
    # ===== Prediction Task Parameters =====
    seq_len=96,
    label_len=48,  # label_len = seq_len//2
    pred_len=96,
    enc_in=137,

    # ===== Patching Related =====
    patch_len=32,
    stride=4,
    padding_patch='end',

    # ===== Moving Average Related =====
    ma_type='ema',
    alpha=0.39,
    beta=0.39,

    # ===== Optimizer and Training =====
    num_workers=5,
    itr=1,
    train_epochs=50,
    batch_size=32,
    patience=10,
    learning_rate=0.0001,

    # ===== LTSF Model Specific Parameters =====
    num_heads=4,
    layers2=['m'],
    hidden_size=32,
    fc1_size=16,
    fc2_size=137,  # Should equal enc_in
    revin=False,
    revin_affine=True,
    revin_subtract_last=False,

    # ===== GPU Related Parameters =====
    use_gpu=True,
    gpu=0,
    use_multi_gpu=False,
    devices='0,1,2,3',
)

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

Exp = Exp_Main

if args.is_training:
    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.des, ii)

        exp = Exp(args)
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)

        torch.cuda.empty_cache()
else:
    ii = 0
    setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_{}_{}'.format(args.model_id,
                                                          args.model,
                                                          args.data,
                                                          args.features,
                                                          args.seq_len,
                                                          args.label_len,
                                                          args.pred_len,
                                                          args.des, ii)

    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)
    torch.cuda.empty_cache()