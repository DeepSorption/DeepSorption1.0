import csv
import numpy as np
import pandas as pd
from mendeleev import element
import matplotlib.pyplot as plt

import torch
from torch.nn.utils.rnn import pad_sequence
from scipy.interpolate import interp1d


def co2_interpolate(path_y='../raw_data/EXPMOF/CO2/MOF_CO2_Uptake.csv', draw=False):
    with open(path_y, "r") as f:
        reader = csv.reader(f, delimiter=",")
        next(reader)
        co2 = {}
        for x in reader:
            name = x[-1]
            if name not in co2.keys():
                co2[name] = [[float(x[2]), float(x[3])]]
            else:
                co2[name].append([float(x[2]), float(x[3])])

    for i, j in co2.items():
        if len(j) == 1:
            continue
        j = np.array(j)
        f = interp1d(j[:, 0], j[:, 1], kind='linear', fill_value="extrapolate")

        co2[i] = f(np.arange(0.01, 0.98, 0.01))
        if draw:
            xnew = np.arange(min(j[:, 0]), max(j[:, 0]), 0.01)
            ynew = f(xnew)
            plt.plot(j[:, 0], j[:, 1], 'o', xnew, ynew, '-')
            plt.show()
    torch.save(co2, '../pre_data/co2/exp_co2.pt')

def read_co2(path_x='../raw_data/EXPMOF/CO2/car/', path_y='../raw_data/EXPMOF/CO2/MOF_CO2_features.csv', path_co2='../pre_data/co2/exp_co2.pt'):
    co2_dict = torch.load(path_co2)
    label = pd.read_csv(path_y)
    column_name = label.columns
    data_x, data_y, data_c = [], [], []

    for name in co2_dict.keys():
        name2 = name.replace('·', '_').rstrip() + '.car'
        print(f'processing at {name}...')

        co2 = torch.tensor(co2_dict[name]).unsqueeze(0)
        prop = torch.tensor(label[label['MOF_name']==name2][column_name[2:]].to_numpy())
        if prop.shape[0] == 0:
            print(f'    {name2} not exist in property file')
            continue
        try:
            df = pd.read_csv(path_x + name2, delimiter='\s+', header=None, skiprows=5, skipfooter=2,
                             engine='python', converters={7: lambda x: element(x).atomic_number})
            df_c = pd.read_csv(path_x + name.rstrip() + '.car', delimiter='\s+', header=None, skiprows=4, nrows=1,
                               engine='python')
        except:
            print(f'    error occurred when reading {name2}')
            continue
        data_x.append(torch.tensor(df[[1, 2, 3, 7]].to_numpy()))
        data_y.append(torch.cat((co2, prop), dim=1))
        data_c.append(torch.tensor(df_c[[0, 1, 2]].to_numpy()))

    data = [data_x, torch.cat(data_y).float(), torch.cat(data_c)]
    torch.save(data, '../pre_data/co2/co2.pt')
    print('Data loading finished with {} samples.'.format(len(data_x)))



if __name__ == '__main__':
    co2_interpolate()
    read_co2()
