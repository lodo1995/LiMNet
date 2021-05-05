import glob
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from util import TimedStep, pandas2torch

class PacketDataset(torch.utils.data.Dataset):
    def __init__(self, Xe, labels):
        super().__init__()
        self.Xe = Xe
        self.labels = labels

    def __getitem__(self, id):
        return self.Xe[id], tuple(lbls[id] for lbls in self.labels)

    def __len__(self):
        return self.Xe.shape[0]

class DataLoader:

    ALL_FEATURES = [
        'eth_dst_1','eth_dst_2','eth_dst_3','eth_src_1','eth_src_2','eth_src_3','eth_type',
        'ip_version','ip_hdr_len','ip_dsfield','ip_len','ip_id','ip_flags','ip_frag_offset','ip_ttl','ip_proto','ip_checksum','ip_src_1','ip_src_2','ip_dst_1','ip_dst_2',
        'port_src','port_dst','tcp_seq_1','tcp_seq_2','tcp_ack_1','tcp_ack_2','tcp_hdr_len','tcp_reserved','tcp_flags','tcp_window_size_value','tcp_checksum','tcp_urgent_pointer'
    ]
    PORT_AND_SIZE_ONLY = [
        'ip_len', 'port_src','port_dst'
    ]
    ALL_FEATURES_EXCEPT_IDS = [
        'eth_type',
        'ip_version','ip_hdr_len','ip_dsfield','ip_len','ip_id','ip_flags','ip_frag_offset','ip_ttl','ip_proto','ip_checksum',
        'port_src','port_dst','tcp_seq_1','tcp_seq_2','tcp_ack_1','tcp_ack_2','tcp_hdr_len','tcp_reserved','tcp_flags','tcp_window_size_value','tcp_checksum','tcp_urgent_pointer'
    ]
    ALL_FEATURES_LIMITED_IDS = [
        'eth_type',
        'ip_version','ip_hdr_len','ip_dsfield','ip_len','ip_id','ip_flags','ip_frag_offset','ip_ttl','ip_proto','ip_checksum','ip_src_1','ip_dst_1',
        'port_src','port_dst','tcp_seq_1','tcp_seq_2','tcp_ack_1','tcp_ack_2','tcp_hdr_len','tcp_reserved','tcp_flags','tcp_window_size_value','tcp_checksum','tcp_urgent_pointer'
    ]

    sequenceLength = 1

    dtypes = {
        'eth_dst_1': np.uint16,
        'eth_dst_2': np.uint16,
        'eth_dst_3': np.uint16,
        'eth_src_1': np.uint16,
        'eth_src_2': np.uint16,
        'eth_src_3': np.uint16,
        'eth_type': np.uint16,
        'ip_version': np.uint16,
        'ip_hdr_len': np.uint16,
        'ip_dsfield': np.uint16,
        'ip_len': np.uint16,
        'ip_id': np.uint16,
        'ip_flags': np.uint16,
        'ip_frag_offset': np.uint16,
        'ip_ttl': np.uint16,
        'ip_proto': np.uint16,
        'ip_checksum': np.uint16,
        'ip_src_1': np.uint16,
        'ip_src_2': np.uint16,
        'ip_dst_1': np.uint16,
        'ip_dst_2': np.uint16,
        'tcp_srcport': np.uint16,
        'tcp_dstport': np.uint16,
        'tcp_seq_1': np.uint16,
        'tcp_seq_2': np.uint16,
        'tcp_ack_1': np.uint16,
        'tcp_ack_2': np.uint16,
        'tcp_hdr_len': np.uint16,
        'tcp_reserved': np.uint16,
        'tcp_flags': np.uint16,
        'tcp_window_size_value': np.uint16,
        'tcp_checksum': np.uint16,
        'tcp_urgent_pointer': np.uint16,
        'malware': object,
        'traffic_type': object,
        'behaviour': object,
        'device_type': object,
    }

    def __init__(self, paths, edgeFeatures, reduceTokens = True):
        self.paths = paths
        self.edgeFeatures = edgeFeatures
        self.reduceTokens = reduceTokens

    def loadData(self, tasks):
        files = tqdm([file for path in self.paths for file in glob.iglob(path)], desc = 'loading data')
        data = pd.concat((pd.read_csv(file, dtype = self.dtypes) for file in files), ignore_index = True, copy = False)

        with TimedStep('Fixing data issues'):
            data['port_src'] = data['tcp_srcport']
            data['port_dst'] = data['tcp_dstport']
            data.drop(columns = ['tcp_srcport', 'tcp_dstport'], inplace = True)
            data.fillna(0, inplace = True)

        with TimedStep('Computing labels'):
            num_outputs = max(task.output for task in tasks) + 1
            labels = tuple([] for _ in range(num_outputs))
            for task in tasks:
                labels[task.output].append(task.compute_labels(data, None, None))
            labels = tuple(torch.stack(lbls, dim = -1) for lbls in labels)

        with TimedStep('Computing edge features'):
            Xe = data[self.edgeFeatures]
            if self.reduceTokens:
                id2tok = pd.unique(Xe.values.ravel('K'))
                tok2id = {int(tok): id for id, tok in enumerate(id2tok)}
                Xe = Xe.applymap(tok2id.get)
                Ntoks = len(id2tok)
            else:
                Ntoks = 2**16
                id2tok = range(Ntoks)
                tok2id = {int(tok): id for id, tok in enumerate(id2tok)}
            Xe = pandas2torch(Xe, dtype=np.int32)

        info = {
            'E': Xe.shape[0],
            'Fe': Xe.shape[1],
            'Ntoks': Ntoks,
            'id2tok': [int(tok) for tok in id2tok],
            'tok2id': tok2id
        }

        return PacketDataset(Xe, labels), info