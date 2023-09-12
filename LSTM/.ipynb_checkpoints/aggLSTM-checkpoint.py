import os
import time
import re
import argparse
import numpy
import torch
import torch.nn as nn


parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=str, default='0')
parser.add_argument("--lr", type=float, default=0.0004)
parser.add_argument("--dataset", type=str, default='.')
parser.add_argument("--emb", type=int, default=40)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--kpl", type=str, default='')
parser.add_argument("--label", type=str, default='range(7)')
parser.add_argument("--atoms", type=int, default='10', help='周围最近原子数')
args = parser.parse_args()
# Device configuration
device = torch.device('cuda:'+args.gpu if torch.cuda.is_available() else 'cpu')
label = eval(args.label)
# Hyper-parameters
hidden_size = 128
num_layers = 2
num_classes = len(label)
vocab_size = 100  # 原子种类 1~100
embed_size = args.emb   # 原子embedded
batch_size = 1
num_epochs = args.epochs
input_size = embed_size + 4
L2 = 3e-4  # L2正则化
init_epoch = 0

if args.kpl:
    init_epoch = int(re.search('-(\d+)emb', args.kpl).groups()[0])
    num_epochs += init_epoch

print('embsize\tlearning_rate\n{}\t{}'.format(embed_size, args.lr))


class atomDataset(torch.utils.data.Dataset):
    def __init__(self, type):
        self.list = crystal_split[type]

    def __getitem__(self, index):
        # item = self.item[index]
        cry = crystal_data[self.list[index]]
        lab = crystal_label[self.list[index]]
        y = numpy.array([(lab[i]-ave[i])/std[i] for i in range(7)], dtype=numpy.float32)
        return torch.tensor(cry[:, :args.atoms*3+6]), torch.tensor(y[label])  # 用最近三个6+3*3

    def __len__(self):
        return len(self.list)


class aggLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, vocab_size, embed_size):
        super(aggLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.emb = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc0 = nn.Linear((embed_size+2)*(args.atoms+1)-1, embed_size)
        self.fc1 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)  # 2 for bidirection
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        x1 = torch.cat((x[0, :, 4].unsqueeze(1), self.emb.weight[x[0, :, 5].long()]), dim=-1)
        for i in range(args.atoms):
            x1 = torch.cat((x1, x[0, :, (i*3+6):(i*3+8)], self.emb.weight[x[0, :, 8+i*3].long()]), dim=-1)
        x2 = self.fc0(x1)
        x3 = torch.cat((x[0, :, :4], x2), dim=-1).unsqueeze(0)
        out, _ = self.lstm(x3, (h0, c0))

        # Decode the hidden state of the last time step
        out = self.fc1(out[:, -1, :])
        return out

    def save(self):
        numpy.save('emb_x.npy', self.emb.weight.detach().cpu().numpy())


crystal_data = numpy.load(args.dataset+'/dataset.npy', allow_pickle=True)
crystal_label = numpy.load(args.dataset+'/labels.npy', allow_pickle=True)
crystal_split = numpy.load(args.dataset+'/split.ple', allow_pickle=True)
# 平均值，用于归一化
ave = crystal_split['AVE']
std = crystal_split['STD']
print('Loading data finished')
train_dataset = atomDataset('train')
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = atomDataset('test')
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, shuffle=False)

model = aggLSTM(input_size, hidden_size, num_layers, num_classes, vocab_size, embed_size).to(device)

if args.kpl:
    model.load_state_dict(torch.load(args.kpl))

exps = f'{args.dataset}/{args.label}-{args.atoms}'
if not os.path.exists(exps):
    os.mkdir(exps)
# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # , weight_decay=L2
print(time.strftime('%m/%d-%H:%M', time.localtime(time.time()+8*3600)))


def train():
    LOSS = 0
    for i, (crystal, labels) in enumerate(train_loader):
        crystal = crystal.to(device)
        labels = labels.to(device)
        outputs = model(crystal)
        loss = criterion(outputs, labels)
        LOSS += float(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 100 == 0 or i == total_step-1:
            print('\rEPOCH:{}/{} STEP:{:.0f}/{:.0f} LOSS:{:.5f}'.format(epoch+1, num_epochs, i/100, total_step/100, LOSS/(i+1)), end=' ')
    TRAINLOSS = 'train{:.5f}'.format(LOSS/total_step)
    return TRAINLOSS


def test(TRAINLOSS, epoch):
    with torch.no_grad():
        LOSS = 0
        f = open(exps+'/'+TRAINLOSS+'mse_all_emb' + str(embed_size), 'w', encoding='utf-8')
        f.write('属性\t预测\t实际\n')
        for i, (crystal, labels) in enumerate(test_loader):
            crystal = crystal.to(device)
            labels = labels.to(device)
            outputs = model(crystal)
            loss = criterion(outputs, labels)
            LOSS += loss.item()
            for j, k in enumerate(label):
                f.write('{}\t{:.4f}\t{:.4f}\n'.format(k, outputs.view(-1)[j].item(), labels.view(-1)[j].item()))

        #print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
        total = len(test_loader)
        print('      TEST LOSS:{:.5f}'.format(LOSS/total))
        f.close()
    TESTLOSS = 'test{:.5f}'.format(LOSS/total)
    torch.save(model.state_dict(), f'{exps}/{TRAINLOSS}-{TESTLOSS}-{epoch}emb{embed_size}.ckpt')


# Train the model
total_step = len(train_loader)
for epoch in range(init_epoch, num_epochs):
    TRAINLOSS = train()
    test(TRAINLOSS, epoch)
