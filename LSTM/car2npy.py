# %%
import pandas
import numpy
import pickle
import torch
import random
import torch.nn.functional as F
from math import cos, sin, sqrt, pi
from tqdm import tqdm

entries = pandas.read_csv('attribute.csv', header=0)
entries = entries.fillna(value=0)  # 将nan填充为0
vender = {1: 0.108, 2: 0.134, 3: 0.175, 4: 0.205, 5: 0.147, 6: 0.149, 7: 0.141, 8: 0.14, 9: 0.139, 10: 0.168, 11: 0.184, 12: 0.205, 13: 0.211, 14: 0.207, 15: 0.192, 16: 0.182, 17: 0.183, 18: 0.193, 19: 0.205, 20: 0.221, 21: 0.216, 22: 0.187, 23: 0.179, 24: 0.189, 25: 0.197, 26: 0.194, 27: 0.192, 28: 0.184, 29: 0.186, 30: 0.21, 31: 0.208, 32: 0.215, 33: 0.206, 34: 0.193, 35: 0.198, 36: 0.212, 37: 0.216, 38: 0.224, 39: 0.219, 40: 0.186, 41: 0.207, 42: 0.209, 43: 0.209, 44: 0.207, 45: 0.195, 46: 0.202, 47: 0.203, 48: 0.23, 49: 0.236, 50: 0.233,
          51: 0.225, 52: 0.223, 53: 0.223, 54: 0.221, 55: 0.222, 56: 0.251, 57: 0.24, 58: 0.235, 59: 0.239, 60: 0.229, 61: 0.236, 62: 0.229, 63: 0.233, 64: 0.237, 65: 0.221, 66: 0.229, 67: 0.216, 68: 0.235, 69: 0.227, 70: 0.242, 71: 0.221, 72: 0.212, 73: 0.217, 74: 0.21, 75: 0.217, 76: 0.216, 77: 0.202, 78: 0.209, 79: 0.217, 80: 0.209, 81: 0.235, 82: 0.232, 83: 0.243, 84: 0.229, 85: 0.236, 86: 0.243, 87: 0.256, 88: 0.243, 89: 0.26, 90: 0.237, 91: 0.243, 92: 0.24, 93: 0.221, 94: 0.256, 95: 0.256, 96: 0.256, 97: 0.256, 98: 0.256, 99: 0.256, 100: 0.256}
atom2idx = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60,
            'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100, 'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110, 'Rg': 111, 'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117, 'Og': 118}
invaldatom = [[9999, 9999, 9999, 9999, 0, 0]]*10
labelName = list(entries.columns)


def expand(crystal, length, tengle):
    x, y, z = length
    a, b, r = [al/180*pi for al in tengle]
    crystal = numpy.array(crystal)
    vect = [[x, 0, 0, 0],
            [y*cos(r), y*sin(r), 0, 0],
            [z*cos(b), z*(cos(a)-cos(b)*cos(r))/sin(r), 0, 0]]
    vect[2][2] = sqrt(z**2-vect[2][0]**2-vect[2][1]**2)
    vect = numpy.array(vect)
    fin = []
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            for k in [-1, 0, 1]:
                fin += (crystal + vect[0]*i + vect[1]*j + vect[2]*k).tolist()
    return fin


crystals = []
labels = []
sideLength = []
angle = []
name = []
for _, cry in tqdm(entries.iterrows(), total=len(entries), desc='Loading'):
    crystal = []
    labels.append([float(x) for x in cry[labelName[2:]]])
    name.append(cry["filename"])
    with open(f'car/{cry["filename"]}.cif.car', 'r') as f:
        for i, line in enumerate(f):
            if i < 4 or len(line) < 40:
                continue
            if i == 4:
                Lattice_parameters = line.strip().split()
                sideLength.append([float(x) for x in Lattice_parameters[1:4]])
                angle.append([float(x) for x in Lattice_parameters[4:7]])
            else:
                _, x, y, z, _, _, _, t, _ = line.strip().split()
                crystal.append([float(x), float(y), float(z), atom2idx[t]])
    spand_cry = torch.tensor(expand(crystal, sideLength[-1], angle[-1])).cuda()
    crystal = torch.tensor(crystal).cuda()
    x = []
    for atom in crystal:
        dis = F.pairwise_distance(atom[:3].repeat(len(spand_cry), 1), spand_cry[:, :3], p=2)
        dis, idxs = torch.sort(dis, descending=False)
        atom = atom.tolist()
        tmp = atom[:3] + [sqrt(atom[0]**2+atom[1]**2+atom[2]**2), vender[atom[3]], atom[3]]
        for i in range(1, 11):
            tmp += [dis[i].item(), vender[spand_cry[idxs[i]][3].item()], spand_cry[idxs[i]][3].item()]
        tmp[0], tmp[1], tmp[2] = tmp[0]**2, tmp[1]**2, tmp[2]**2
        x.append(tmp)
    x.sort(key=(lambda x: x[3]))
    crystals.append(numpy.array(x, dtype=numpy.float32))

# %%
cnt = len(labels)
labels = numpy.array(labels)
allList = list(range(cnt))
random.shuffle(allList)
split = {'test': allList[:int(cnt*0.05)], 'valid': allList[int(cnt*0.05):int(cnt*0.05)*2], 'train': allList[int(cnt*0.05)*2:]}
split['AVE'] = labels[split['train']].mean(0)
split['STD'] = labels[split['train']].std(0)
print('MEAN\n', split['AVE'])
print('STD\n', split['STD'])


numpy.save('dataset.npy', numpy.array(crystals, dtype=object))
numpy.save('labels.npy', labels)
numpy.save('sideLength.npy', numpy.array(sideLength))
numpy.save('angle.npy', numpy.array(angle))
with open('name.ple', 'wb') as f:
    pickle.dump(name, f)
with open('split.ple', 'wb') as f:
    pickle.dump(split, f)
