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
from Matformer.msa_np import build_model
from utils.utils import parse_args, Logger, set_seed
import pdb

def main():
    args = parse_args()
    args.save_path='save/'
    args.gpu_id='0,1,2,3'
    log = Logger(args.save_path + 'hmof/', f'msa_{strftime("%Y-%m-%d_%H-%M-%S", localtime())}.log')
    set_seed(args.seed)
    args.batch_size = 64
    args.dist_bar = [[3, 5, 8, 1e10]]


    target_dict = ['LCD', 'PLD', 'LFPD', 'D', 'ASA', 'ASA', 'NASA', 'NASA', 'AVVF', 'AV', 'NAV', 'CO2', 'CO2']
    tgt = [0, 1, 3, 5, 8, 9, 12]
   

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
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
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
    
    train_size = data_x_train.shape[0]
    dev_size = data_x_val.shape[0]
    test_size = data_x_test.shape[0]
    
    criterion = torch.nn.MSELoss()
    log.logger.info('{} hMOF {}\nMax_len: {}; Train: {}; Dev: {}; Test: {}\nTarget: {}\nGPU: {}; Batch_size: {}; lr: {}; fnn_dim: {}'
                    .format("=" * 40, "=" * 40, data_x_train.shape[1], train_size, dev_size, test_size,
                            [target_dict[i] for i in tgt], args.gpu_id, args.batch_size, args.lr, args.fnn_dim))


    log.logger.info('{} Start Training- Dist: {} {}'.format("=" * 40, args.dist_bar, "=" * 40))
    best_loss, best_mse = 1e9, 1e9
    t0 = time()
    early_stop = 0

    model = build_model(args.atom_class + 1, len(tgt), args.dist_bar).cuda()
    if len(args.gpu_id) > 1:  model = torch.nn.DataParallel(model)
    optimizer = opt.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = opt.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.6, patience=10, min_lr=5e-7)

    for epoch in range(0, 100):
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

            with torch.no_grad(): pred = model(dev_atom, dev_mask, dev_pos)
            mse += mse_loss(pred[:, 0], labels[:, 0], reduction='sum').item() / test_size * scales[0][1]

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
        if early_stop >= 10:
            log.logger.info(f'Early Stopping!!! No Improvement on Loss for 20 Epochs.')
            break
        lr_scheduler.step(mse)
    log.logger.info('{} End Training (Time: {:.2f}h) {}'.format("=" * 40, (time() - t0) / 3600, "=" * 40))
    log.logger.info('Dist_bar: {} | Best Epoch: {} | Dev_MSE: {:.2f}'.format(args.dist_bar, best_epoch, best_mse))
    checkpoint = {'model': best_model.state_dict(), 'n_tgt': len(tgt), 'dist_bar': args.dist_bar,
                    'epochs': best_epoch, 'lr': optimizer.param_groups[0]['lr'],'scales':scales}
    if len(args.gpu_id) > 1: checkpoint['model'] = best_model.module.state_dict()
    torch.save(checkpoint, args.save_path + 'hmof/' + 'hmof_np_8layer_Tgt_fnndim{}_lr{}_batchsize{}.pt'.format(args.fnn_dim, args.lr, args.batch_size))
    log.logger.info('Save the best model as hmof_Tgt_{}_{}_{}.pt'.format(args.fnn_dim, args.lr, args.batch_size))

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

    test_pred[:, 0] = relu(test_pred[:, 0] * scales[0][1] + scales[0][0])
    test_label[:, 0] = test_label[:, 0] * scales[0][1] + scales[0][0]
    r2 = round(r2_score(test_label[:, 0].cpu().numpy(), test_pred[:, 0].cpu().numpy()), 3)
    log.logger.info(f'Test R2: {r2}\n\n')


if __name__ == '__main__':
    main()
