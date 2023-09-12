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
    
def main():
    args = parse_args()
    save_path='save/'
    log = Logger(save_path + 'coremof_new/', f'msa_{strftime("%Y-%m-%d_%H-%M-%S", localtime())}.log')
    
 
    set_seed(args.seed)

    args.dist_bar = [[3, 5, 8, 1e10]]
    args.fnn_dim=2048
    args.batch_size=8
    args.dropout=0.15
    args.embed_dim=512


     
    target_dict = ['LCD', 'PLD', 'LFPD', 'D', 'ASA', 'ASA', 'NASA', 'NASA', 'AVVF', 'AV', 'NAV', 'CO2', 'CO2']
    target = [0, 1, 3, 5, 8, 9, 12]
 
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
  
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
    
    train_size=len(data_x_train)
    val_size=len(data_x_val)
    test_size=len(data_x_test)
    criterion = torch.nn.MSELoss()
    log.logger.info('{} CORE-MOF_expand {}\nMax_len: {}; Train: {}; Dev: {}; Test: {}\nTarget: {}\nGPU: {}; Batch_size: {}; lr: {}; fnn_dim: {}; embed_dim: {}; dropout: {}'
                    .format("=" * 40, "=" * 40, 0, len(data_x_train), len(data_x_val), len(data_x_test),
                            [target_dict[i] for i in target], args.gpu_id, args.batch_size, args.lr, args.fnn_dim, args.embed_dim, args.dropout))

    
    log.logger.info('{} Start Training- Dist: {} {}'.format("=" * 40, args.dist_bar, "=" * 40))
    best_loss, best_mse = 1e9, 1e9
    t0 = time()
    early_stop = 0

    model=build_model(vocab=(args.atom_class + 1), tgt=len(target), dist_bar=args.dist_bar, N=6, embed_dim=args.embed_dim, ffn_dim=args.fnn_dim,dropout=args.dropout).cuda()
    print('model_build_successful!')
    if len(args.gpu_id) > 1:  model = torch.nn.DataParallel(model)
    optimizer = opt.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = opt.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.7, patience=10, min_lr=5e-7)

    for epoch in range(0,500):
        model.train()
        loss = 0.0
        t1 = time()

        for crystal, labels,dist in train_loader:
            dist=dist.cuda()
            crystal=crystal.cuda()
            labels=labels.cuda()# crystal [8, 1024, 4]
            crystal_atom = crystal[..., 3].long()   # [8, 1024]
            crystal_pos = crystal[..., :3]  # [8, 1024, 3]
            crystal_mask = (crystal_atom != 0).unsqueeze(1) 
            pred = model(crystal_atom, crystal_mask, crystal_pos,dist)
            loss_batch = criterion(pred, labels)
            loss += loss_batch.item()
            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()
        del pred
        torch.cuda.empty_cache()

        model.eval()
        mse = 0
        for crystal, labels,dist in dev_loader:
            crystal=crystal.cuda()
            labels=labels.cuda()
            dist=dist.cuda()
            dev_atom = crystal[..., 3].long()
            dev_mask = (dev_atom != 0).unsqueeze(1)
            dev_pos = crystal[..., :3]
            

           
            with torch.no_grad(): pred = model(dev_atom, dev_mask, dev_pos,dist)
            mse += mse_loss(pred[:, -1], labels[:, -1], reduction='sum').item() / test_size * scales[-1][1]

        if mse < best_mse:
            best_mse = deepcopy(mse)
            best_model = deepcopy(model)
            best_epoch = epoch + 1
        if loss < best_loss:
            best_loss = deepcopy(loss)
            early_stop = 0
        else:
            early_stop += 1
        log.logger.info('Epoch: {} | Time: {:.1f}s | Loss: {:.2f} | MSE: {:.2f} | Lr: {:.1f}'.
                        format(epoch + 1, time() - t1, loss, mse, optimizer.param_groups[0]['lr'] * 1e5))
        if early_stop >= 40:
            log.logger.info(f'Early Stopping!!! No Improvement on Loss for 20 Epochs.')
            break
        lr_scheduler.step(mse)
        test_pred, test_label = [], []
        best_model.eval()
        for crystal, labels,dist in test_dataset:
            dist=dist.cuda()
            crystal=crystal.cuda()
            labels=labels.cuda()
            test_atom = crystal[:, :, 3].long()
            test_mask = (test_atom != 0).unsqueeze(1)
            test_pos = crystal[:, :, :3]

            with torch.no_grad():
                pred = best_model(test_atom, test_mask, test_pos,dist)
                test_pred.append(pred)
                test_label.append(labels)
        test_pred = torch.cat(test_pred)
        test_label = torch.cat(test_label)

        test_pred[:, -1] = relu(test_pred[:, -1] * scales[-1][1] + scales[-1][0])
        test_label[:, -1] = test_label[:, -1] * scales[-1][1] + scales[-1][0]
        r2 = round(r2_score(test_label[:, -1].cpu().numpy(), test_pred[:, -1].cpu().numpy()), 3)
        log.logger.info(f'Test R2: {r2}\n\n')
        
    log.logger.info('{} End Training (Time: {:.2f}h) {}'.format("=" * 40, (time() - t0) / 3600, "=" * 40))
    log.logger.info('Dist_bar: {} | Best Epoch: {} | Dev_MSE: {:.2f}'.format(args.dist_bar, best_epoch, best_mse))
    checkpoint = {'model': best_model.state_dict(), 'n_tgt': len(target), 'dist_bar': args.dist_bar,
                    'epochs': best_epoch, 'lr': optimizer.param_groups[0]['lr'],'scales':scales}
    if len(args.gpu_id) > 1: checkpoint['model'] = best_model.module.state_dict()
    torch.save(checkpoint, save_path + 'coremof_new/' + 'Coremof_newpos_Tgt_aug_fnndim{}_embeddim{}_lr{}_batchsize{}_dropout{}.pt'.format(args.fnn_dim,args.embed_dim, args.lr, args.batch_size, args.dropout))
    log.logger.info('Save the best model as Coremof_newpos_Tgt_fnndim{}_embeddim{}_lr{}_batchsize{}_dropout{}.pt'.format(args.fnn_dim,args.embed_dim, args.lr, args.batch_size, args.dropout))
if __name__ == '__main__':
    main()