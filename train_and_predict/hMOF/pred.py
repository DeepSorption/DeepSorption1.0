import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn.utils import shuffle
from sklearn.metrics import r2_score, mean_squared_error
import os
import math
import sys
from torch.utils.data import DataLoader, TensorDataset
sys.path.append("..")
from Matformer.msa import build_model
from utils.utils import parse_args, set_seed
import pdb

args = parse_args()
set_seed(args.seed)


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
cpt_name = '../save/hmof/hmof_model.pt'
cpt = torch.load(cpt_name)
model = build_model(args.atom_class + 1, cpt['n_tgt'], cpt['dist_bar']).cuda()
model.load_state_dict(cpt['model'])
model.eval()
print('Model Loading Successfully. Start Testing!')





data_all_train=torch.load('../pre_data/hmof/hmof_train.pt')
data_all_val=torch.load('../pre_data/hmof/hmof_val.pt')
data_all_test=torch.load('../pre_data/hmof/hmof_test.pt')
    
data_x_train = pad_sequence(data_all_train[0], batch_first=True, padding_value=0)
data_x_val = pad_sequence(data_all_val[0], batch_first=True, padding_value=0) 
data_x_test = pad_sequence(data_all_test[0], batch_first=True, padding_value=0) 
    
data_y_train = torch.tensor([i.numpy() for i in (data_all_train[1])]).float()
data_y_val = torch.tensor([i.numpy() for i in (data_all_val[1])]).float()
data_y_test = torch.tensor([i.numpy() for i in (data_all_test[1])]).float()
    
data_c_train=torch.tensor([i.numpy() for i in (data_all_train[2])]).float()/2
data_c_val=torch.tensor([i.numpy() for i in (data_all_val[2])]).float()/2
data_c_test=torch.tensor([i.numpy() for i in (data_all_test[2])]).float()/2
    
data_c_train = torch.cat((data_c_train, torch.ones(data_c_train.shape[0]).unsqueeze(1).double() * args.atom_class), dim=1)   
    data_x_train = torch.cat((data_c_train.unsqueeze(1), data_x_train), dim=1)
    data_c_val = torch.cat((data_c_val, torch.ones(data_c_val.shape[0]).unsqueeze(1).double() * args.atom_class), dim=1)   
data_x_val = torch.cat((data_c_val.unsqueeze(1), data_x_val), dim=1) 
data_c_test = torch.cat((data_c_test, torch.ones(data_c_test.shape[0]).unsqueeze(1).double() * args.atom_class), dim=1)   
data_x_test = torch.cat((data_c_test.unsqueeze(1), data_x_test), dim=1) 
    
    
data_x_train, data_y_train = data_x_train[:, :args.max_len].cuda(), data_y_train.cuda()
data_x_val, data_y_val = data_x_val[:, :args.max_len].cuda(), data_y_val.cuda()
data_x_test, data_y_test = data_x_test[:, :args.max_len].cuda(), data_y_test.cuda()
    
scales = [[data_y_train[:, i].mean().item(), data_y_train[:, i].std().item()] for i in range(data_y_train.shape[-1])]
for i in range(data_y_train.shape[-1]):
    data_y_train[:, i] = (data_y_train[:, i] - scales[i][0]) / scales[i][1]
    data_y_val[:, i] = (data_y_val[:, i] - scales[i][0]) / scales[i][1]
    data_y_test[:, i] = (data_y_test[:, i] - scales[i][0]) / scales[i][1]

train_dataset=TensorDataset(data_x_train, data_y_train)
dev_dataset=TensorDataset(data_x_val, data_y_val)
test_dataset=TensorDataset(data_x_test, data_y_test)
    
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size * 2)
test_dataset = DataLoader(test_dataset, batch_size=args.batch_size * 2)   

print('Testing on the training set.')
train_pred, train_label = [], []
for crystal, labels in train_loader:
    with torch.no_grad():
        crystal_atom = crystal[..., 3].long()   # [8, 1024]
        crystal_pos = crystal[..., :3]  # [8, 1024, 3]
        crystal_mask = (crystal_atom != 0).unsqueeze(1) 
        pred = model(crystal_atom, crystal_mask, crystal_pos)

        train_pred.append(pred)
        train_label.append(labels)# crystal [8, 1024, 4]
            
            

train_pred = torch.cat(train_pred)
train_label = torch.cat(train_label)
for i in range(train_pred.shape[-1]):
    train_pred[:, i] = train_pred[:, i] * scales[i][1] + scales[i][0]
train_r2 = round(r2_score(train_label[:, 0].cpu().numpy(), train_pred[:, 0].cpu().numpy()), 3)
train_rmse = round(mean_squared_error(train_label[:, 0].cpu().numpy(), train_pred[:, 0].cpu().numpy()), 3)
np.save('../save/hmof/MOF_train_multi_16.npy', [train_label.cpu().numpy(), train_pred.cpu().numpy()])
del train_pred
del train_label
torch.cuda.empty_cache()
print(f'Train Size: {train_size}, Train R2: {train_r2}; Train RMSE: {train_rmse}')

print('Testing on the dev set.')
dev_pred, dev_label = [], []
for crystal, labels in dev_loader:
    with torch.no_grad():
        dev_atom = crystal[..., 3].long()   # [8, 1024]
        dev_pos = crystal[..., :3]  # [8, 1024, 3]
        dev_mask = (dev_atom != 0).unsqueeze(1) 
        pred = model(dev_atom, dev_mask, dev_pos)
        dev_pred.append(pred)
        dev_label.append(labels)

dev_pred = torch.cat(dev_pred)
dev_label = torch.cat(dev_label)
for i in range(dev_pred.shape[-1]):
    dev_pred[:, i] = dev_pred[:, i] * scales[i][1] + scales[i][0]
dev_r2 = round(r2_score(dev_label[:, 0].cpu().numpy(), dev_pred[:, 0].cpu().numpy()), 3)
dev_rmse = round(mean_squared_error(dev_label[:, 0].cpu().numpy(), dev_pred[:, 0].cpu().numpy()), 3)
np.save('../save/hmof/MOF_dev_multi_16.npy', [dev_label.cpu().numpy(), dev_pred.cpu().numpy()])
del dev_label
del dev_pred
torch.cuda.empty_cache()
print(f'Dev Size: {dev_size}; Dev R2: {dev_r2}; DEV RMSE: {dev_rmse}')

print('Testing on the test set.')
test_pred, test_label = [], []
for crystal, labels in test_loader:
    with torch.no_grad():
        test_atom = crystal[..., 3].long()   # [8, 1024]
        test_pos = crystal[..., :3]  # [8, 1024, 3]
        test_mask = (crystal_atom != 0).unsqueeze(1) 
        pred = model(test_atom, test_mask, test_pos)
 
        test_pred.append(pred)
        test_label.append(labels)

test_pred = torch.cat(test_pred)
test_label = torch.cat(test_label)
for i in range(test_pred.shape[-1]):
    test_pred[:, i] = test_pred[:, i] * scales[i][1] + scales[i][0]
test_r2 = round(r2_score(test_label[:, 0].cpu().numpy(), test_pred[:, 0].cpu().numpy()), 3)
test_rmse = round(mean_squared_error(test_label[:, 0].cpu().numpy(), test_pred[:, 0].cpu().numpy()), 3)

np.save('../save/hmof/MOF_test_multi_16.npy', [test_label.cpu().numpy(), test_pred.cpu().numpy()])
print(f'Test Size: {test_size}; Test R2: {test_r2}; Test RMSE: {test_rmse}')

