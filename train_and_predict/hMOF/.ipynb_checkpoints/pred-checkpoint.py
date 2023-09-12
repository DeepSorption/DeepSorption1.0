import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn.utils import shuffle
from sklearn.metrics import r2_score, mean_squared_error
import os
import math
import sys

sys.path.append("..")
from Matformer.msa import build_model
from utils.utils import parse_args, set_seed
import pdb

args = parse_args()
set_seed(args.seed)


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
cpt_name = '../save/mof/MOF_Tgt_7_16.pt'
cpt = torch.load(cpt_name)
model = build_model(args.atom_class + 1, cpt['n_tgt'], cpt['dist_bar']).cuda()
model.load_state_dict(cpt['model'])
model.eval()
print('Model Loading Successfully. Start Testing!')

data_x = torch.load('../pre_data/mof/mof_x.pt')
data_x = [i[:args.max_len] for i in data_x]
target_dict = ['CO2', 'N2', 'CO2/N2', 'LFSA', 'LISAFSPD', 'Surface_Area', 'Void_Fraction']
tgt = [i for i in range(len(target_dict))]
# tgt = [0]
print(f'Testing Targets: {[target_dict[i] for i in tgt]}')
data_y = torch.load('../pre_data/mof/mof_y_old.pt').float()[:, tgt]
data_c = torch.load('../pre_data/mof/mof_c.pt') / 2
data_c = torch.cat((data_c, torch.ones(data_c.shape[0]).unsqueeze(1).double() * args.atom_class), dim=1)
data_c, data_y = data_c.cuda(), data_y.cuda()
data_x, data_y, data_c = shuffle(data_x, data_y, data_c, random_state=0)
train_size = int(args.split_ratio * len(data_x))
dev_size = (len(data_x) - train_size) // 2
test_size = len(data_x) - train_size - dev_size
train_batches = math.ceil(train_size / args.batch_size)
dev_batches = math.ceil(dev_size / args.batch_size)
test_batches = math.ceil(test_size / args.batch_size)
scales = [[data_y[:, i].mean().item(), data_y[:, i].std().item()] for i in range(data_y.shape[-1])]

print('Testing on the training set.')
train_pred, train_label = [], []
for batch in range(train_batches):
    crystal_x = data_x[batch * args.batch_size: (batch + 1) * args.batch_size]
    crystal_x = pad_sequence(crystal_x, batch_first=True, padding_value=0).cuda()
    crystal_c = data_c[batch * args.batch_size: (batch + 1) * args.batch_size]
    crystal = torch.cat((crystal_c.unsqueeze(1), crystal_x), dim=1)
    labels = data_y[batch * args.batch_size: (batch + 1) * args.batch_size]

    crystal_atom = crystal[:, :, 3].long()
    crystal_pos = crystal[:, :, :3]
    crystal_mask = (crystal_atom != 0).unsqueeze(1)
    with torch.no_grad():
        pred = model(crystal_atom, crystal_mask, crystal_pos)
        train_pred.append(pred)
        train_label.append(labels)

train_pred = torch.cat(train_pred)
train_label = torch.cat(train_label)
for i in range(train_pred.shape[-1]):
    train_pred[:, i] = train_pred[:, i] * scales[i][1] + scales[i][0]
train_r2 = round(r2_score(train_label[:, 0].cpu().numpy(), train_pred[:, 0].cpu().numpy()), 3)
train_rmse = round(mean_squared_error(train_label[:, 0].cpu().numpy(), train_pred[:, 0].cpu().numpy()), 3)
np.save('../save/mof/MOF_train_multi_16.npy', [train_label.cpu().numpy(), train_pred.cpu().numpy()])
del train_pred
del train_label
torch.cuda.empty_cache()
print(f'Train Size: {train_size}, Train R2: {train_r2}; Train RMSE: {train_rmse}')

print('Testing on the dev set.')
dev_pred, dev_label = [], []
for batch in range(dev_batches):
    crystal_x = data_x[train_size:][batch * args.batch_size: (batch + 1) * args.batch_size]
    crystal_x = pad_sequence(crystal_x, batch_first=True, padding_value=0).cuda()
    crystal_c = data_c[train_size:][batch * args.batch_size: (batch + 1) * args.batch_size]
    crystal = torch.cat((crystal_c.unsqueeze(1), crystal_x), dim=1)
    labels = data_y[train_size:][batch * args.batch_size: (batch + 1) * args.batch_size]

    dev_atom = crystal[:, :, 3].long()
    dev_mask = (dev_atom != 0).unsqueeze(1)
    dev_pos = crystal[:, :, :3]

    with torch.no_grad():
        pred = model(dev_atom, dev_mask, dev_pos)
        dev_pred.append(pred)
        dev_label.append(labels)

dev_pred = torch.cat(dev_pred)
dev_label = torch.cat(dev_label)
for i in range(dev_pred.shape[-1]):
    dev_pred[:, i] = dev_pred[:, i] * scales[i][1] + scales[i][0]
dev_r2 = round(r2_score(dev_label[:, 0].cpu().numpy(), dev_pred[:, 0].cpu().numpy()), 3)
dev_rmse = round(mean_squared_error(dev_label[:, 0].cpu().numpy(), dev_pred[:, 0].cpu().numpy()), 3)
np.save('../save/mof/MOF_dev_multi_16.npy', [dev_label.cpu().numpy(), dev_pred.cpu().numpy()])
del dev_label
del dev_pred
torch.cuda.empty_cache()
print(f'Dev Size: {dev_size}; Dev R2: {dev_r2}; DEV RMSE: {dev_rmse}')

print('Testing on the test set.')
test_pred, test_label = [], []
for batch in range(test_batches):
    crystal_x = data_x[train_size + dev_size:][batch * args.batch_size: (batch + 1) * args.batch_size]
    crystal_x = pad_sequence(crystal_x, batch_first=True, padding_value=0).cuda()
    crystal_c = data_c[train_size + dev_size:][batch * args.batch_size: (batch + 1) * args.batch_size]
    crystal = torch.cat((crystal_c.unsqueeze(1), crystal_x), dim=1)
    labels = data_y[train_size + dev_size:][batch * args.batch_size: (batch + 1) * args.batch_size]

    test_atom = crystal[:, :, 3].long()
    test_mask = (test_atom != 0).unsqueeze(1)
    test_pos = crystal[:, :, :3]

    with torch.no_grad():
        pred = model(test_atom, test_mask, test_pos)
        test_pred.append(pred)
        test_label.append(labels)

test_pred = torch.cat(test_pred)
test_label = torch.cat(test_label)
for i in range(test_pred.shape[-1]):
    test_pred[:, i] = test_pred[:, i] * scales[i][1] + scales[i][0]
test_r2 = round(r2_score(test_label[:, 0].cpu().numpy(), test_pred[:, 0].cpu().numpy()), 3)
test_rmse = round(mean_squared_error(test_label[:, 0].cpu().numpy(), test_pred[:, 0].cpu().numpy()), 3)

np.save('../save/mof/MOF_test_multi_16.npy', [test_label.cpu().numpy(), test_pred.cpu().numpy()])
print(f'Test Size: {test_size}; Test R2: {test_r2}; Test RMSE: {test_rmse}')

