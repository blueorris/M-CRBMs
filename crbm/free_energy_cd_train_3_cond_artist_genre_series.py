import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

import numpy as np
from numpy.random import default_rng
import pickle
from tqdm import tqdm
import os
from pathlib import Path
import shutil
import click
import pandas as pd

import config
from free_energy_cd_model_3_cond import CRBM
import utils as u
import val_and_test as vt


########## CONFIGURATION ##########
HIDDEN_UNITS = config.HIDDEN_UNITS
BATCH_SIZE = config.BATCH_SIZE
EPOCHS = config.EPOCHS
LEARNING_RATE = config.LEARNING_RATE
WEIGHT_DECAY = config.WEIGHT_DECAY
K = config.CD_K
OPTIM = config.OPTIM
CATEGORY = config.CATEGORY
CUDA = torch.cuda.is_available()
print('CUDA available:', CUDA)

# create save path
save_path = './exp/crbm_free_energy_3_cond/hin_{0}_eps_{1}_optim_{2}_lr_{3}_wd_{4}_cate_{5}/'.format(HIDDEN_UNITS, EPOCHS, OPTIM, LEARNING_RATE, WEIGHT_DECAY, CATEGORY)
if not os.path.isdir(save_path):
    Path(save_path).mkdir(parents=True, exist_ok=True)
else:
    print("this parameter combination has already been tested")
    click.confirm('Do you want to overwrite old param?', abort=True)
    shutil.rmtree(save_path)


########## LOAD DATASET ##########
print('Loading training & test dataset...')
# training dataset
train_main_arr = np.load('./data/dataset_train_main.npy')
train_artist_arr = np.load('./data/dataset_train_artist.npy')
train_genre_arr = np.load('./data/dataset_train_genre.npy')
train_series_arr = np.load('./data/dataset_train_series_word.npy')
# test dataset
test_main_arr = np.load('./data/dataset_test_main.npy')
test_artist_arr = np.load('./data/dataset_test_artist.npy')
test_genre_arr = np.load('./data/dataset_test_genre.npy')
test_series_arr = np.load('./data/dataset_test_series_word.npy')

print('Shape of train_main_arr:', train_main_arr.shape)
print('Shape of train_artist_arr:', train_artist_arr.shape)
print('Shape of train_genre_arr:', train_genre_arr.shape)
print('Shape of train_series_arr:', train_series_arr.shape)

print('Shape of test_main_arr:', test_main_arr.shape)
print('Shape of test_artist_arr:', test_artist_arr.shape)
print('Shape of test_genre_arr:', test_genre_arr.shape)
print('Shape of test_series_arr:', test_series_arr.shape)


########## FILL DATA LOADER ##########
# training data loader
train_main_tensor = torch.from_numpy(train_main_arr).float()
train_main_loader = torch.utils.data.DataLoader(train_main_tensor, shuffle=False, batch_size=BATCH_SIZE, drop_last=True)

train_artist_tensor = torch.from_numpy(train_artist_arr).float()
train_artist_loader = torch.utils.data.DataLoader(train_artist_tensor, shuffle=False, batch_size=BATCH_SIZE, drop_last=True)

train_genre_tensor = torch.from_numpy(train_genre_arr).float()
train_genre_loader = torch.utils.data.DataLoader(train_genre_tensor, shuffle=False, batch_size=BATCH_SIZE, drop_last=True)

train_series_tensor = torch.from_numpy(train_series_arr).float()
train_series_loader = torch.utils.data.DataLoader(train_series_tensor, shuffle=False, batch_size=BATCH_SIZE, drop_last=True)

# test data loader
test_main_tensor = torch.from_numpy(test_main_arr).float()
test_main_loader = torch.utils.data.DataLoader(test_main_tensor, shuffle=False, batch_size=BATCH_SIZE, drop_last=True)

test_artist_tensor = torch.from_numpy(test_artist_arr).float()
test_artist_loader = torch.utils.data.DataLoader(test_artist_tensor, shuffle=False, batch_size=BATCH_SIZE, drop_last=True)

test_genre_tensor = torch.from_numpy(test_genre_arr).float()
test_genre_loader = torch.utils.data.DataLoader(test_genre_tensor, shuffle=False, batch_size=BATCH_SIZE, drop_last=True)

test_series_tensor = torch.from_numpy(test_series_arr).float()
test_series_loader = torch.utils.data.DataLoader(test_series_tensor, shuffle=False, batch_size=BATCH_SIZE, drop_last=True)


########## INITIAL CRBM ##########
VISIBLE_UNITS = len(train_main_arr[0])
COND_UNITS_1 = len(train_artist_arr[0])
COND_UNITS_2 = len(train_genre_arr[0])
COND_UNITS_3 = len(train_series_arr[0])

crbm = CRBM(VISIBLE_UNITS, COND_UNITS_1, COND_UNITS_2, COND_UNITS_3, HIDDEN_UNITS, K, use_cuda=CUDA)
if CUDA:
    crbm = crbm.cuda()

########## TRAINING CRBM ##########
print('Training CRBM...')

# choose optimizor
if OPTIM == 'adam':
    train_op = optim.Adam(crbm.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
elif OPTIM == 'rms':
    train_op = optim.RMSprop(crbm.parameters())

# create lists and dicts to save training status
epoch_error = []
epoch_loss = []
history_mean_test_recall = {'top_1':[], 'top_5':[], 'top_10':[], 'top_20':[]}
history_mean_test_precision = {'top_1':[], 'top_5':[], 'top_10':[], 'top_20':[]}
history_mean_test_f1_score = {'top_1':[], 'top_5':[], 'top_10':[], 'top_20':[]}

# training phase
for epoch in range(EPOCHS):
    curr_error_ = []
    curr_loss_ = []

    print('Epoch=%d' % (epoch+1))
    pbar = tqdm(total=len(train_main_loader))

    for i, batch in enumerate(zip(train_main_loader, train_artist_loader, train_genre_loader, train_series_loader)):
        main_batch = batch[0]
        artist_batch = batch[1]
        genre_batch = batch[2]
        series_batch = batch[3]
        if CUDA:
            main_batch = main_batch.cuda()
            artist_batch = artist_batch.cuda()
            genre_batch = genre_batch.cuda()
            series_batch = series_batch.cuda()
        batch_error, batch_loss = crbm.get_model_loss(main_batch, artist_batch, genre_batch, series_batch)
        curr_error_.append(batch_error.item())
        curr_loss_.append(batch_loss.item())

        train_op.zero_grad()
        batch_loss.backward()
        train_op.step() # update the parameters
        pbar.update(1)
    pbar.close()

    mean_curr_error_ = np.mean(curr_error_)
    mean_curr_loss_ = np.mean(curr_loss_)
    print('Epoch Error: %.4f' % (mean_curr_error_))
    print('Epoch loss: %.4f' % (mean_curr_loss_))
    epoch_error.append(mean_curr_error_)
    epoch_loss.append(mean_curr_loss_)

    # create dicts to save test results
    test_recall_list = {'top_1':[], 'top_5':[], 'top_10':[], 'top_20':[]}
    test_precision_list = {'top_1':[], 'top_5':[], 'top_10':[], 'top_20':[]}
    test_f1_score_list = {'top_1':[], 'top_5':[], 'top_10':[], 'top_20':[]}

    for _, test_batch in enumerate(zip(test_main_loader, test_artist_loader, test_genre_loader, test_series_loader)):
        test_main_batch = test_batch[0]
        test_artist_batch = test_batch[1]
        test_genre_batch = test_batch[2]
        test_series_batch = test_batch[3]
        if CUDA:
            test_main_batch = test_main_batch.cuda()
            test_artist_batch = test_artist_batch.cuda()
            test_genre_batch = test_genre_batch.cuda()
            test_series_batch = test_series_batch.cuda()
        test_v_p, test_v_act = crbm.forward(test_main_batch, test_artist_batch, test_genre_batch, test_series_batch)

        test_recall_list, test_precision_list, test_f1_score_list = vt.val(test_main_batch, test_v_p, test_recall_list, test_precision_list, test_f1_score_list, CUDA, EPOCHS, epoch, save_path)

    # print(test_recall_list, test_precision_list)
    epoch_mean_test_recall = {}
    epoch_mean_test_precision = {}
    epoch_mean_test_f1_score = {}
    # mean of recall
    for k in test_recall_list.keys():
        epoch_mean_test_recall[k] = np.mean(test_recall_list[k])
        history_mean_test_recall[k].append(np.mean(test_recall_list[k]))
    # mean precision
    for k in test_precision_list.keys():
        epoch_mean_test_precision[k] = np.mean(test_precision_list[k])
        history_mean_test_precision[k].append(np.mean(test_precision_list[k]))
    # mean f1 score
    for k in test_f1_score_list.keys():
        epoch_mean_test_f1_score[k] = np.mean(test_f1_score_list[k])
        history_mean_test_f1_score[k].append(np.mean(test_f1_score_list[k]))
    # print the results
    for k in test_recall_list.keys():
        print('epoch mean')
        print(k)
        print('recall: %.4f, precision: %.4f, f1_value: %.4f' % (epoch_mean_test_recall[k], epoch_mean_test_precision[k], epoch_mean_test_f1_score[k]))


##########  SAVE  ##########
# save error, loss, val recall, val precision.
# np.save(save_path + 'epoch_error.npy', epoch_error)
# np.save(save_path + 'epoch_loss.npy', epoch_loss)
# np.save(save_path + 'history_mean_val_recall.npy', history_mean_val_recall)
# np.save(save_path + 'history_mean_val_precision.npy', history_mean_val_precision)
save_df = pd.DataFrame()
save_df['his_error'] = epoch_error
save_df['his_loss'] = epoch_loss
# top 1
save_df['top_1_recall'] = history_mean_test_recall['top_1']
save_df['top_1_precision'] = history_mean_test_precision['top_1']
save_df['top_1_f1'] = history_mean_test_f1_score['top_1']
# top 5
save_df['top_5_recall'] = history_mean_test_recall['top_5']
save_df['top_5_precision'] = history_mean_test_precision['top_5']
save_df['top_5_f1'] = history_mean_test_f1_score['top_5']
# top 10
save_df['top_10_recall'] = history_mean_test_recall['top_10']
save_df['top_10_precision'] = history_mean_test_precision['top_10']
save_df['top_10_f1'] = history_mean_test_f1_score['top_10']
# top 20
save_df['top_20_recall'] = history_mean_test_recall['top_20']
save_df['top_20_precision'] = history_mean_test_precision['top_20']
save_df['top_20_f1'] = history_mean_test_f1_score['top_20']

save_df.to_csv(save_path + "metrics.csv")

# save model
torch.save(crbm.state_dict(), save_path + '/model.pth')
print('Model saved.')
