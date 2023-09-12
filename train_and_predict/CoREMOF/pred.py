import os
from time import time, strftime, localtime

import torch
import torch.optim as opt
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import mse_loss, relu
from sklearn.metrics import r2_score
from copy import deepcopy
import sys

sys.path.append("..")
from Matformer.msa_dist import build_model
from utils.utils import parse_args, Logger, set_seed
import pdb
from pymatgen.core.structure import Structure
import numpy as np
import pymatgen

args = parse_args()
set_seed(args.seed)

import time 
time_start=time.time()
def gen_super_structure(cif):
    
    struc=Structure.from_file(cif)
    coords=struc.cart_coords
    supersize=get_super(coords.shape[0])
    struc.make_supercell(supersize)
    
    dist=struc.distance_matrix
    return dist

def get_structure(cif):
    
    struc=Structure.from_file(cif)
    
    
    dist=struc.distance_matrix
    return dist

def get_super(i):
    size=[1,1,1]
    if i<=38:
        size=[3,3,3]
    elif i>38 and i<=57:
        size=[3,3,2]
    elif i>57 and i<=86:
        size=[3,2,2]
    elif i>86 and i<=128:
        size=[2,2,2]
    elif i>128 and i<=256:
        size=[2,2,1]
    elif i>256 and i<=512:
        size=[2,1,1]
    return size


class atomDataset(torch.utils.data.Dataset):
    def __init__(self, data_x,data_y,data_dist):
        self.data_x = data_x
        self.data_y = data_y
        self.data_dist = data_dist

    def __getitem__(self, index):
 
        cry = self.data_x[index]
        dist = self.data_dist[index]
        lab = self.data_y[index,:]
        return cry,lab,dist  

    def __len__(self):
        return len(self.data_x)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
cpt_name = '.../save/coremof/CoREMOF_model.pt'
cpt = torch.load(cpt_name)

tgt = [0, 1, 3, 5, 8, 9, 12]
model = build_model(args.atom_class + 1, cpt['n_tgt'], cpt['dist_bar']).cuda()
model.load_state_dict(cpt['model'])
model.eval()
args.batch_size = 16
print('Model Loading Successfully. Start Testing!')

seednum=34
data_all=torch.load('../pre_data/coremof/coremof_super_all.pt')
data_all_std=torch.load('../pre_data/coremof/coremof_all.pt')
a=np.arange(len(data_all[0]))
train_ratio=0.7
val_ratio=0.15
test_ratio=0.15
    
np.random.seed(seednum)
np.random.shuffle(a)
train_idx=a[:int(len(a)*train_ratio)]
val_idx=a[int(len(a)*train_ratio):-int(len(a)*test_ratio)]
test_idx=a[-int(len(a)*test_ratio):]
    


#for super
data_x_all=data_all[0]
data_c_all=torch.tensor([i.numpy() for i in (data_all[2])]).float()/2
data_c_all = torch.cat((data_c_all, torch.ones(data_c_all.shape[0]).unsqueeze(1).double() * args.atom_class), dim=1)   
data_x_all =[torch.cat((data_c_all[i,:].unsqueeze(0), data_x_all[i]), dim=0) for i in range(len(data_x_all))]

data_x_train = [data_x_all[i] for i in train_idx]
data_x_val = [data_x_all[i] for i in val_idx]
data_x_test = [data_x_all[i] for i in test_idx]
    
data_y_all = torch.tensor([i.numpy() for i in (data_all[1])]).float()
data_y_train=data_y_all[train_idx]
data_y_val=data_y_all[val_idx]
data_y_test=data_y_all[test_idx]
    
data_cif_all=data_all[5]
dist_all=[]
    
f=0
for cif in data_cif_all:
    if f%100==0:
        print(f)
    dist=gen_super_structure(os.path.join('../pre_data/coremof/coremof', cif+'.cif'))
    dist=np.row_stack((np.zeros(dist.shape[0]),dist))
    dist=np.column_stack((np.zeros(dist.shape[0]),dist))
    dist_all.append(dist)
    f+=1

data_dist_train=[dist_all[i] for i in train_idx]
data_dist_val=[dist_all[i] for i in val_idx]
data_dist_test=[dist_all[i] for i in test_idx]
    
scales = [[data_y_train[:, i].mean().item(), data_y_train[:, i].std().item()] for i in range(data_y_train.shape[-1])]
for i in range(data_y_train.shape[-1]):
    data_y_train[:, i] = (data_y_train[:, i] - scales[i][0]) / scales[i][1]
    data_y_val[:, i] = (data_y_val[:, i] - scales[i][0]) / scales[i][1]
    data_y_test[:, i] = (data_y_test[:, i] - scales[i][0]) / scales[i][1]
        
#for std
data_x_all_std=data_all_std[0]
data_c_all_std=torch.tensor([i.numpy() for i in (data_all_std[2])]).float()/2
data_c_all_std = torch.cat((data_c_all_std, torch.ones(data_c_all_std.shape[0]).unsqueeze(1).double() * args.atom_class), dim=1)   
data_x_all_std =[torch.cat((data_c_all_std[i,:].unsqueeze(0), data_x_all_std[i]), dim=0) for i in range(len(data_x_all_std))]

data_x_train_std = [data_x_all_std[i] for i in train_idx]
    
    
data_y_all_std = torch.tensor([i.numpy() for i in (data_all_std[1])]).float()
data_y_train_std=data_y_all_std[train_idx]
    
    
data_cif_all_std=data_all_std[5]
data_cif_all_std_train=[data_cif_all_std[i] for i in train_idx]
dist_all_std_train=[]
    
f=0
for cif in data_cif_all_std_train:
    if f%100==0:
        print(f)
    dist=get_structure(os.path.join('../pre_data/coremof/coremof', cif+'.cif'))
    dist=np.row_stack((np.zeros(dist.shape[0]),dist))
    dist=np.column_stack((np.zeros(dist.shape[0]),dist))
    dist_all_std_train.append(dist)
    f+=1

    
    

for i in range(data_y_train_std.shape[-1]):
    data_y_train_std[:, i] = (data_y_train_std[:, i] - scales[i][0]) / scales[i][1]
        
data_x_train=data_x_train+data_x_train_std
del data_x_train_std
data_y_train=torch.cat((data_y_train,data_y_train_std))
data_dist_train=data_dist_train+dist_all_std_train   

train_dataset=atomDataset(data_x_train, data_y_train,data_dist_train)
dev_dataset=atomDataset(data_x_val, data_y_val,data_dist_val)
test_dataset=atomDataset(data_x_test, data_y_test,data_dist_test)
    
train_loader = DataLoader(train_dataset, batch_size=1,shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=1)
test_dataset = DataLoader(test_dataset, batch_size=1)

print('Testing on the training set.')
train_pred, train_label = [], []
for crystal, labels,dist in train_loader:        
    dist=dist.cuda()
    crystal=crystal.cuda()
    labels=labels.cuda()# crystal [8, 1024, 4]
    crystal_atom = crystal[..., 3].long()   # [8, 1024]
    crystal_pos = crystal[..., :3]  # [8, 1024, 3]
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
for crystal, labels,dist in dev_loader:        
    dist=dist.cuda()
    crystal=crystal.cuda()
    labels=labels.cuda()# crystal [8, 1024, 4]
    crystal_atom = crystal[..., 3].long()   # [8, 1024]
    crystal_pos = crystal[..., :3]  # [8, 1024, 3]
    crystal_mask = (crystal_atom != 0).unsqueeze(1) 
    with torch.no_grad():
        pred = model(crystal_atom, crystal_mask, crystal_pos)
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
for crystal, labels,dist in test_loader:        
    dist=dist.cuda()
    crystal=crystal.cuda()
    labels=labels.cuda()# crystal [8, 1024, 4]
    crystal_atom = crystal[..., 3].long()   # [8, 1024]
    crystal_pos = crystal[..., :3]  # [8, 1024, 3]
    crystal_mask = (crystal_atom != 0).unsqueeze(1) 
    with torch.no_grad():
        pred = model(crystal_atom, crystal_mask, crystal_pos)
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