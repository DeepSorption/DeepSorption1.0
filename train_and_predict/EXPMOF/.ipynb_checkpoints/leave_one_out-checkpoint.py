import os
from time import time, strftime, localtime

import torch
import torch.optim as opt
from torch.nn.functional import relu, mse_loss
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score

import sys
sys.path.append('..')
from Matformer.msa import build_model
from utils.utils import parse_args, Logger, set_seed


def main():
    args = parse_args()
    expmof = args.expmof
    log = Logger('../' + args.save_path + f'{expmof}/', f'base_{strftime("%Y-%m-%d_%H-%M-%S", localtime())}.log')
    set_seed(args.seed)

    data_x, data_y, data_c = torch.load(f'../pre_data/{expmof}/{expmof}.pt')
    data_x = pad_sequence(data_x, batch_first=True, padding_value=0)
    data_x = data_x[:, :args.max_len]
    if expmof =='c2h2':
        target = [expmof + '_{:.2f}'.format(0.06 + 0.01 * i) for i in range(89)] + ['pore1', 'pore2', 'vol', 'd', 'asa']
    else:
        target = [expmof + '_{:.2f}'.format(0.01 + 0.01 * i) for i in range(97)] + ['pore1', 'pore2', 'vol', 'd', 'asa']
    scales = [[data_y[:, i].mean().item(), data_y[:, i].std().item()] for i in range(data_y.shape[-1])]
    for i in range(data_y.shape[-1]):
        data_y[:, i] = (data_y[:, i] - scales[i][0]) / scales[i][1]
    data_c = torch.cat((data_c, torch.ones(data_c.shape[0]).unsqueeze(1).double() * args.atom_class), dim=1)
    data_x = torch.cat((data_c.unsqueeze(1), data_x), dim=1)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    data_x, data_y = data_x.cuda(), data_y.cuda()

    criterion = torch.nn.MSELoss()
    log.logger.info('{} LOO {}\nTrain: {}; Test: {}; GPU: {}; Epochs: {}; Batch Size: {}'
                    .format("=" * 20, "=" * 20, len(data_x) - 1, 1, args.gpu_id, 100, 1))
    loo = LeaveOneOut()
    log.logger.info('{} Start Training {}'.format("=" * 20, "=" * 20))
    loo_mse = torch.zeros(data_y.shape[0]).cuda()
    loo_pred, loo_label = torch.zeros_like(data_y).cuda(), torch.zeros_like(data_y).cuda()

    for k, (train_idx, test_idx) in enumerate(loo.split(data_x.cpu())):
        log.logger.info('{} Start {}-fold Training {}'.format("*" * 20, k + 1, "*" * 20))
        train_loader = DataLoader(TensorDataset(data_x[train_idx], data_y[train_idx]), batch_size=1, shuffle=True)
        test_loader = DataLoader(TensorDataset(data_x[test_idx], data_y[test_idx]), batch_size=1, shuffle=False)

        best_loss, best_mse = 1e9, 1000
        early_stop = 0
        model = build_model(args.atom_class + 1, len(target), args.dist_bar).cuda()
        optimizer = opt.Adam(model.parameters(), lr=0.0005, weight_decay=1e-7)
        lr_scheduler = opt.lr_scheduler.LambdaLR(optimizer, lambda epo: 0.9 ** epo)

        diff = set(torch.unique(data_x[test_idx][..., 3].long()).cpu().numpy()) -\
               set(torch.unique(data_x[train_idx][..., 3].long()).cpu().numpy())
        for epoch in range(0, 100):
            loss, n_aug = 0.0, 0
            model.train()
            t1 = time()
            for crystal, labels in train_loader:
                crystal_atom = crystal[..., 3].long()
                crystal_pos = crystal[..., :3]
                crystal_mask = (crystal_atom != 0).unsqueeze(1)
                crystal_pos[:, 0, :3] /= 2

                pred = model(crystal_atom, crystal_mask, crystal_pos)
                loss_batch = criterion(pred, labels)
                loss += loss_batch.item()
                optimizer.zero_grad()
                loss_batch.backward()
                optimizer.step()

            model.eval()
            mse = 0
            for crystal, labels in test_loader:
                with torch.no_grad():
                    test_atom = crystal[:, :, 3].long()
                    test_mask = (test_atom != 0).unsqueeze(1)
                    test_pos = crystal[:, :, :3]
                    test_pos[:, 0, :3] /= 2
                    pred = model(test_atom, test_mask, test_pos)
                    mse = mse_loss(pred[:, :-5], labels[:, :-5])

            if best_mse > mse:
                best_mse = mse
                best_pred = pred
            if loss < best_loss:
                best_loss = loss
                early_stop = 0
            else:
                early_stop += 1
            log.logger.info('Epoch: {} | Time: {:.1f}s | Loss: {:.2f} | MSE: {:.2f} | Lr: {:.2f}'.
                            format(epoch + 1, time() - t1, loss, mse, optimizer.param_groups[0]['lr'] * 10000))
            if early_stop >= 20:
                log.logger.info('Early Stopping!!! No Improvement on Loss for 20 Epochs.\n')
                break
            lr_scheduler.step()
        if len(diff) == 0: diff = 'No difference'
        log.logger.info(f'Element difference: {diff}')
        loo_mse[test_idx] = best_mse
        loo_label[test_idx] = data_y[test_idx]
        loo_pred[test_idx] = best_pred

    for i in range(loo_pred.shape[-1]):
        loo_pred[:, i] = relu(loo_pred[:, i] * scales[i][1] + scales[i][0])
        loo_label[:, i] = loo_label[:, i] * scales[i][1] + scales[i][0]
    r2 = round(r2_score(loo_label[:, :-5].reshape(-1).cpu().numpy(), loo_pred[:, :-5].reshape(-1).cpu().numpy()), 3)
    log.logger.info('Leave-One-Out R2: {}'.format(r2))
    torch.save([loo_pred, loo_label, loo_mse], f'../save/expmof/loo_{expmof}.pt')


if __name__ == '__main__':
    main()
