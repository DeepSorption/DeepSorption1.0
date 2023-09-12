import numpy as np
import pandas as pd
from mendeleev import element
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from scipy.interpolate import interp1d
import csv

def c2h2_interpolate(path_y='../raw_data/EXPMOF/C2H2/MOF_C2H2_Uptake.csv', draw=False):
    with open(path_y, "r") as f:
        reader = csv.reader(f, delimiter=",")
        next(reader)

        c2h2 = {}
        for x in reader:
            name = x[-1]

            if name not in c2h2.keys():
                c2h2[name] = [[float(x[2]), float(x[3])]]
            else:
                c2h2[name].append([float(x[2]), float(x[3])])

    for i, j in c2h2.items():
        j = np.array(j)
        f = interp1d(j[:, 0], j[:, 1], kind='linear')


        c2h2[i] = f(np.arange(0.06, 0.95, 0.01))
        if draw:
            xnew = np.arange(min(j[:, 0]), max(j[:, 0]), 0.01)
            ynew = f(xnew)
            plt.plot(j[:, 0], j[:, 1], 'o', xnew, ynew, '-')
            plt.show()
    torch.save(c2h2, '../pre_data/c2h2/c2h2_interp.pt')


def read_c2h2(path_x='../raw_data/EXPMOF/C2H2/car/', path_y='../raw_data/EXPMOF/C2H2/MOF_C2H2_features.csv', path_c2h2='../pre_data/c2h2/c2h2_interp.pt'):
    c2h2_dict = torch.load(path_c2h2)
    label = pd.read_csv(path_y)
    column_name = label.columns
    data_x, data_y, data_c = [], [], []

    for name in c2h2_dict.keys():
        name2 = name.replace('·', '_').rstrip() + '.car'
        print(f'processing at {name}...')

        co2 = torch.tensor(c2h2_dict[name]).unsqueeze(0)
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
    torch.save(data, '../pre_data/c2h2/c2h2.pt')
    print('Data loading finished with {} samples.'.format(len(data_x)))


if __name__ == '__main__':
    c2h2_interpolate()
    read_c2h2()







