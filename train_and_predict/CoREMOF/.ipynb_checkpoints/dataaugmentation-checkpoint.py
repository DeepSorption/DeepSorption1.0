import os
from time import time, strftime, localtime

import torch
import torch.optim as opt
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import mse_loss, relu
from sklearn.metrics import r2_score
import numpy as np
import sys
from copy import deepcopy
import copy
#sys.path.append("..")
from msa_np import build_model
from utils import parse_args, Logger, set_seed
import pdb
from torch import cos, sin
def data_trans(trans_ratio,test_data_x,center,angle):
    pos_new=[]
    if trans_ratio==0:
        return test_data_x

    for j in range(len(test_data_x)):
        n_conformer=[test_data_x[j][:,:3].max()*trans_ratio]
        for translation in n_conformer:
            x_tmp = test_data_x[j][:,:3]+ translation
            a, b, c = center[j, 0], center[j, 1], center[j, 2]
            alpha, beta, gamma = angle[j, 0]/180*np.pi, angle[j, 1]/180*np.pi, angle[j, 2]/180*np.pi

            tmp1 = c * torch.sqrt(sin(beta) ** 2 - (cos(alpha) - cos(beta) * cos(gamma)) ** 2 / sin(gamma) ** 2)
            tmp2 = c * (cos(alpha) - cos(beta) * cos(gamma)) / sin(gamma)
            tmp3 = c * cos(beta)

            for i in range(1):
                m = x_tmp[:,  2] // tmp1
                x_tmp[:,  2] -= m * tmp1
                x_tmp[:,  1] -= m * tmp2
                x_tmp[:, 0] -= m * tmp3

            for i in range(1):
                n = x_tmp[:,  2] / tmp1
                x_tmp[:,  1] -= ((x_tmp[:,  1] - n * tmp2) // (b * sin(gamma))) * (b * sin(gamma))
                x_tmp[:, 0] -= ((x_tmp[:,  1] - n * tmp2) // (b* sin(gamma))) * (b * cos(gamma))
                x_tmp[:, 0] -= (x_tmp[:, 0] - n * tmp3) // a * a

            #x_new.append(x[3])
            pos_new.append(torch.cat((x_tmp,test_data_x[j][:,3:]),axis=1))
            #y_new.append(y[i])

    return pos_new

def cha_angle(x, l):
    ang = copy.deepcopy(x)
    ang[:,:, 0] = x[:,:, l[0]]
    ang[:,:, 1] = x[:,:, l[1]]
    ang[:,:, 2] = x[:,:, l[2]]
    return ang


def main():
    seednum=34
    args = parse_args()
    args.save_path='save_tr_aug/'
    log = Logger(args.save_path + 'coremof_moredistbar/', f'msa_{strftime("%Y-%m-%d_%H-%M-%S", localtime())}_seednum_{seednum}.log')
    set_seed(seednum)
    
    args.batch_size = 64

    args.dist_bar =[[3,5,8,1e10]]
    distbar=[3,5,8,1e10]

    #data_x = torch.load('/home/cjy/cjy/3d-transformer/3D-Transformer/data/coremof_x.pt')
    #data_x = pad_sequence(data_x, batch_first=True, padding_value=0)    # ĺĄŤĺtensorĺ°ç¸ĺéżĺş?[10066, 10560, 4]
    
    target_dict = ['LCD', 'PLD', 'D', 'ASA', 'AVVF', 'AV',  'CO2']
    tgt = [0,1,2,3,4,5,6]
    
    data_all=torch.load('/HOME/scz5707/run/coremof/moreseed/coremof_super_all.pt')
    #data_y = torch.load('/home/cjy/cjy/3d-transformer/3D-Transformer/data/coremof_y.pt').float()
    a=np.arange(len(data_all[0]))
    train_ratio=0.7
    val_ratio=0.15
    test_ratio=0.15
    seednum=34
    np.random.seed(seednum)
    np.random.shuffle(a)
    train_idx=a[:int(len(a)*train_ratio)]
    val_idx=a[int(len(a)*train_ratio):-int(len(a)*test_ratio)]
    test_idx=a[-int(len(a)*test_ratio):]

    torch.save([train_idx,val_idx,test_idx],'index_seed{}.pt'.format(seednum))
    transratio_all=[0,0.5]
    angle_l=[[0,1,2],[0,2,1],[1,2,0],[1,0,2],[2,0,1],[2,1,0]]
    #data_y = torch.cat((data_y[:, tgt], data_gas), dim=1) # [10066, 7]
    data_x_train_all=[]
    data_y_train_all=[]
    data_x_val_all=[]
    data_y_val_all=[]
    data_x_test_all=[]
    data_y_test_all=[]
    for trans_ratio in transratio_all:
        for change_angle in angle_l:
            
            data_x_train = [data_all[0][i] for i in train_idx]
            data_x_val = [data_all[0][i] for i in val_idx]
            data_x_test = [data_all[0][i] for i in test_idx]
    
            data_y_all = torch.tensor([i.numpy() for i in (data_all[1])]).float()
            data_y_train=data_y_all[train_idx]
            data_y_val=data_y_all[val_idx]
            data_y_test=data_y_all[test_idx]

            data_c_all=torch.tensor([i.numpy() for i in (data_all[2])]).float()
            data_c_train=data_c_all[train_idx]
            data_c_val=data_c_all[val_idx]
            data_c_test=data_c_all[test_idx]
            data_angle_all=torch.tensor([i.numpy() for i in (data_all[3])]).float()
            data_angle_train=data_angle_all[train_idx]
            data_angle_val=data_angle_all[val_idx]
            data_angle_test=data_angle_all[test_idx]
            
            data_x_train_new=data_trans(trans_ratio,data_x_train,data_c_train,data_angle_train)
            data_x_train_new = pad_sequence(data_x_train_new, batch_first=True, padding_value=0)
            data_c_train = torch.cat((data_c_train/2, torch.ones(data_c_train.shape[0]).unsqueeze(1).double() * args.atom_class), dim=1)   
            data_x_train_new = torch.cat((data_c_train.unsqueeze(1), data_x_train_new), dim=1)[:, :args.max_len]
            data_x_train_new=cha_angle(data_x_train_new,change_angle)
            data_x_train_all.append(data_x_train_new)
            data_y_train_all.append(data_y_train)
            
            data_x_val_new=data_trans(trans_ratio,data_x_val,data_c_val,data_angle_val)
            data_x_val_new = pad_sequence(data_x_val_new, batch_first=True, padding_value=0)
            data_c_val = torch.cat((data_c_val/2, torch.ones(data_c_val.shape[0]).unsqueeze(1).double() * args.atom_class), dim=1)   
            data_x_val_new = torch.cat((data_c_val.unsqueeze(1), data_x_val_new), dim=1)[:, :args.max_len]
            data_x_val_new=cha_angle(data_x_val_new,change_angle)
            data_x_val_all.append(data_x_val_new)
            data_y_val_all.append(data_y_val)
            
            data_x_test_new=data_trans(trans_ratio,data_x_test,data_c_test,data_angle_test)
            data_x_test_new = pad_sequence(data_x_test_new, batch_first=True, padding_value=0)
            data_c_test = torch.cat((data_c_test/2, torch.ones(data_c_test.shape[0]).unsqueeze(1).double() * args.atom_class), dim=1)   
            data_x_test_new = torch.cat((data_c_test.unsqueeze(1), data_x_test_new), dim=1)[:, :args.max_len]
            data_x_test_new=cha_angle(data_x_test_new,change_angle)
            data_x_test_all.append(data_x_test_new)
            data_y_test_all.append(data_y_test)
    

    data_x_train_final=torch.cat(data_x_train_all)
    data_y_train_final=torch.cat(data_y_train_all)
    data_x_val_final=torch.cat(data_x_val_all)
    data_y_val_final=torch.cat(data_y_val_all)
    data_x_test_final=torch.cat(data_x_test_all)
    data_y_test_final=torch.cat(data_y_test_all)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    data_x_train, data_y_train = data_x_train_final.cuda(), data_y_train_final.cuda()
    data_x_val, data_y_val = data_x_val_final.cuda(), data_y_val_final.cuda()
    data_x_test, data_y_test = data_x_test_final.cuda(), data_y_test_final.cuda()
    
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
    
    train_size = data_x_train.shape[0]
    dev_size = data_x_val.shape[0]
    test_size = data_x_test.shape[0]
 
    criterion = torch.nn.MSELoss()
    log.logger.info('{} CORE-MOF-MORESEED {}\nMax_len: {}; Train: {}; Dev: {}; Test: {}\nTarget: {}\nGPU: {}; Batch_size: {}; distbar: {}'
                    .format("=" * 40, "=" * 40, data_x_train.shape[1], train_size, dev_size, test_size,
                            [target_dict[i] for i in tgt], args.gpu_id, args.batch_size,distbar))


    log.logger.info('{} Start Training- Dist: {} {}'.format("=" * 40, args.dist_bar, "=" * 40))
    best_loss, best_mse = 1e9, 1e9
    t0 = time()
    early_stop = 0

    model = build_model(args.atom_class + 1, len(tgt), args.dist_bar).cuda()
    if len(args.gpu_id) > 1:  model = torch.nn.DataParallel(model)
    optimizer = opt.Adam(model.parameters(), lr=1e-4 * len(args.gpu_id.split(',')))
    lr_scheduler = opt.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.7, patience=10, min_lr=5e-7)

    for epoch in range(0, 300):
        model.train()
        loss = 0.0
        t1 = time()
        for crystal, labels in train_loader:    # crystal [8, 1024, 4]
            crystal_atom = crystal[..., 3].long()   # [8, 1024]
            crystal_pos = crystal[..., :3]  # [8, 1024, 3]
            crystal_mask = (crystal_atom != 0).unsqueeze(1) 
            pred = model(crystal_atom, crystal_mask, crystal_pos)
            loss_batch = criterion(pred, labels)
            loss += loss_batch.item()
            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()
        del pred
        torch.cuda.empty_cache()

        model.eval()
        mse = 0
        for crystal, labels in dev_loader:
            dev_atom = crystal[..., 3].long()
            dev_mask = (dev_atom != 0).unsqueeze(1)
            dev_pos = crystal[..., :3]

            with torch.no_grad(): 
                pred = model(dev_atom, dev_mask, dev_pos)
                mse += mse_loss(pred[:, -1], labels[:, -1], reduction='sum').item() / test_size * scales[-1][1]

        if mse < best_mse:
            best_mse = deepcopy(mse)
            best_model = deepcopy(model)
            best_epoch = epoch + 1
            #early_stop = 0
        if loss < best_loss:
            best_loss = deepcopy(loss)
            early_stop = 0
        else:
            early_stop += 1
        log.logger.info('Seed: {} |Epoch: {} | Time: {:.1f}s | Loss: {:.2f} | MSE: {:.2f} | Lr: {:.1f}'.
                        format(seednum,epoch + 1, time() - t1, loss, mse, optimizer.param_groups[0]['lr'] * 1e5))
        if early_stop >= 40:
            log.logger.info(f'Early Stopping!!! No Improvement on Loss for 20 Epochs.')
            break
        lr_scheduler.step(mse)
    log.logger.info('{} End Training (Time: {:.2f}h) {}'.format("=" * 40, (time() - t0) / 3600, "=" * 40))
    log.logger.info('Dist_bar: {} | Best Epoch: {} | Dev_MSE: {:.2f}'.format(args.dist_bar, best_epoch, best_mse))
    checkpoint = {'model': best_model.state_dict(), 'n_tgt': len(tgt), 'dist_bar': args.dist_bar,
                    'epochs': best_epoch, 'lr': optimizer.param_groups[0]['lr'],'scales':scales}
    if len(args.gpu_id) > 1: checkpoint['model'] = best_model.module.state_dict()
    torch.save(checkpoint,  args.save_path + 'coremof_moredistbar/' + 'Coremof_singletask_layer6_Tgt_{}_batchsize{}_distbar{}_traug.pt'.format(len(tgt), args.batch_size,distbar))
    log.logger.info('Save the best model as Coremof_Tgt_{}_batchsize{}_distbar{}_traug.pt'.format(len(tgt), args.batch_size,distbar))

    test_pred, test_label = [], []
    best_model.eval()
    for crystal, labels in test_dataset:
        test_atom = crystal[:, :, 3].long()
        test_mask = (test_atom != 0).unsqueeze(1)
        test_pos = crystal[:, :, :3]

        with torch.no_grad():
            pred = best_model(test_atom, test_mask, test_pos)
            test_pred.append(pred)
            test_label.append(labels)
    test_pred = torch.cat(test_pred)
    test_label = torch.cat(test_label)

    
    
    for i in range(len(tgt)):
        test_pred[:, i] = relu(test_pred[:, i] * scales[i][1] + scales[i][0])
        test_label[:, i] = test_label[:, i] * scales[i][1] + scales[i][0]
    
    r2 = round(r2_score(test_label[:, -1].cpu().numpy(), test_pred[:, -1].cpu().numpy()), 3)

    np.save('multitaskt358_R2'+str(r2)+'_testresult.npy',np.concatenate((test_label.unsqueeze(0).cpu().numpy(),test_pred.unsqueeze(0).cpu().numpy())))
    log.logger.info(f'Test R2: {r2}\n\n')


if __name__ == '__main__':
    main()