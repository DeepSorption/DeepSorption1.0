import torch
from torch import nn
import pickle
from utils import Triples
import os.path as op
from mendeleev import element
from mendeleev.fetch import fetch_table
from collections import defaultdict as ddict

def fill(name2id, loaded_id2name, emb, loaded_emb):
    for idx, name in loaded_id2name.items():
        real_idx = name2id.get(name, None)
        if real_idx is None:
            continue
        emb[real_idx] = loaded_emb[idx]
    return emb

def get_key (dict, value):
    return [k for k, v in dict.items() if v == value][0]

def fill_atom(name2id, loaded_id2name, emb, loaded_emb):
    for element, atomic in name2id.items():
        loaded_id = get_key(loaded_id2name, element)
        emb[atomic-1] = loaded_emb[loaded_id]
    return emb

def load(file_name, total_entity2id, total_relation2id, e_dim=512, r_dim=512) -> (torch.Tensor, torch.Tensor):
    loaded_dict = pickle.load(open(file_name, 'rb'))
    loaded_relation_emb, loaded_id2relation = loaded_dict['relation'], loaded_dict['id2relation']
    loaded_entity_emb, loaded_id2entity = loaded_dict['entity'], loaded_dict['id2entity']
    
    ptable = fetch_table('elements')
    all_elements = ptable['symbol'].values.tolist()
    all_atomic_number = ptable['atomic_number'].values.tolist()
    element2atomic = dict(zip(all_elements, all_atomic_number))
    
    relation_emb = nn.Embedding(len(total_relation2id), r_dim).weight.data
    entity_emb = nn.Embedding(len(total_entity2id), e_dim).weight.data
    atomic_emb = nn.Embedding(len(element2atomic), e_dim).weight.data

    relation_emb = fill(total_relation2id, loaded_id2relation, relation_emb, loaded_relation_emb)
    entity_emb = fill(total_entity2id, loaded_id2entity, entity_emb, loaded_entity_emb)
    atomic_emb = fill_atom(element2atomic, loaded_id2entity, atomic_emb, loaded_entity_emb)
    
    with open(f'{op.basename(file_name[:-4])}_emb.pkl', 'wb') as f:
        dict_save = {
            'relation_emb': relation_emb,
            'entity_emb': entity_emb,
            'atomic_emb': atomic_emb
        }
        pickle.dump(dict_save, f)
    return relation_emb, entity_emb, atomic_emb



data = Triples()
relation_emb, entity_emb, atomic_emb = load('RotatE_512_256.pkl', total_entity2id=data.entity2id, total_relation2id=data.relation2id, e_dim=512, r_dim=256)

