import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score, mean_squared_error
import os
import sys

sys.path.append("..")
from Matformer.msa import build_model
from utils.utils import parse_args, set_seed
import pdb

args = parse_args()
set_seed(args.seed)

import time 
time_start=time.time()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
cpt_name = '../save/coremof/Coremof_Tgt_7_16.pt'
cpt = torch.load(cpt_name)
# tgt = [12]
tgt = [0, 1, 3, 5, 8, 9, 12]
model = build_model(args.atom_class + 1, cpt['n_tgt'], cpt['dist_bar']).cuda()
model.load_state_dict(cpt['model'])
model.eval()
args.batch_size = 16
print('Model Loading Successfully. Start Testing!')

data_x = torch.load('../pre_data/coremof/coremof_x.pt')
data_x = pad_sequence(data_x, batch_first=True, padding_value=0)
data_y = torch.load('../pre_data/coremof/coremof_y.pt').float()
data_gas = torch.load('../pre_data/coremof/coremof_gas.pt').float()
data_y = torch.cat((data_y, data_gas), dim=1)[:, tgt]

data_c = torch.load('../pre_data/coremof/coremof_c.pt') / 2
data_c = torch.cat((data_c, torch.ones(data_c.shape[0]).unsqueeze(1).double() * args.atom_class), dim=1)
data_x = torch.cat((data_c.unsqueeze(1), data_x), dim=1)
data_x, data_y = data_x[:, :args.max_len].cuda(), data_y.cuda()
scales = [[data_y[:, i].mean().item(), data_y[:, i].std().item()] for i in range(data_y.shape[-1])]

dataset = TensorDataset(data_x, data_y)
train_size = int(args.split_ratio * len(dataset))
dev_size = (len(dataset) - train_size) // 2
test_size = len(dataset) - train_size - dev_size
train_dataset, dev_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, dev_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size)
test_dataset = DataLoader(test_dataset, batch_size=args.batch_size)

print('Testing on the training set.')
train_pred, train_label = [], []
for crystal, labels in train_loader:
    crystal_atom = crystal[..., 3].long()
    crystal_pos = crystal[..., :3]
    crystal_mask = (crystal_atom != 0).unsqueeze(1)
    with torch.no_grad():
        pred = model(crystal_atom, crystal_mask, crystal_pos)
        train_pred.append(pred)
        train_label.append(labels)


train_pred = torch.cat(train_pred)
train_label = torch.cat(train_label)
print(train_pred.shape)
for i in range(train_pred.shape[-1]):
    train_pred[:, i] = train_pred[:, i] * scales[i][1] + scales[i][0]
train_r2 = round(r2_score(train_label[:, -1].cpu().numpy(), train_pred[:, -1].cpu().numpy()), 3)
train_rmse = round(mean_squared_error(train_label[:, -1].cpu().numpy(), train_pred[:, -1].cpu().numpy()), 3)

np.save('../save/coremof/COREMOF_train.npy', [train_label.cpu().numpy(), train_pred.cpu().numpy()])
del train_pred
del train_label
torch.cuda.empty_cache()
print(f'Train Size: {train_size}, Train R2: {train_r2}; Train RMSE: {train_rmse}')


print('Testing on the dev set.')
dev_pred, dev_label = [], []
for crystal, labels in dev_loader:
    dev_atom = crystal[..., 3].long()
    dev_mask = (dev_atom != 0).unsqueeze(1)
    dev_pos = crystal[..., :3]
    with torch.no_grad():
        pred = model(dev_atom, dev_mask, dev_pos)
        dev_pred.append(pred)
        dev_label.append(labels)

dev_pred = torch.cat(dev_pred)
dev_label = torch.cat(dev_label)
for i in range(dev_pred.shape[-1]):
    dev_pred[:, i] = dev_pred[:, i] * scales[i][1] + scales[i][0]
dev_r2 = round(r2_score(dev_label[:, -1].cpu().numpy(), dev_pred[:, -1].cpu().numpy()), 3)
dev_rmse = round(mean_squared_error(dev_label[:, -1].cpu().numpy(), dev_pred[:, -1].cpu().numpy()), 3)
np.save('../save/coremof/COREMOF_dev.npy', [dev_label.cpu().numpy(), dev_pred.cpu().numpy()])
del dev_label
del dev_pred
torch.cuda.empty_cache()
print(f'Dev Size: {dev_size}; Dev R2: {dev_r2}; DEV RMSE: {dev_rmse}')


print('Testing on the test set.')
test_pred, test_label = [], []
for crystal, labels in test_dataset:
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
test_r2 = round(r2_score(test_label[:, -1].cpu().numpy(), test_pred[:, -1].cpu().numpy()), 3)
test_rmse = round(mean_squared_error(test_label[:, -1].cpu().numpy(), test_pred[:, -1].cpu().numpy()), 3)
np.save('../save/coremof/COREMOF_test.npy', [test_label.cpu().numpy(), test_pred.cpu().numpy()])
print(f'Test Size: {test_size}; Test R2: {test_r2}; Test RMSE: {test_rmse}')


time_end=time.time()
print('time cost',time_end-time_start,'s')