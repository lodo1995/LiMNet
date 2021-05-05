import glob
import ipaddress
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from util import TimedStep, pandas2torch, numpy2torch

class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, edgeData, size, starts, Xv):
        super().__init__()
        self.starts = starts
        self.edgeData = edgeData
        self.size = size
        self.Xv = Xv

    def __getitem__(self, id):
        start = self.starts[id]
        stop = start + self.size

        E, Xe, labels = self.edgeData
        E = E[start:stop]
        Xe = Xe[start:stop]
        Xv = self.Xv[E]
        labels = tuple(lbls[start:stop] for lbls in labels)

        V = torch.unique(E)
        I = torch.arange(V.shape[0])
        IM = torch.full((self.Xv.shape[0],), torch.iinfo(torch.int64).max, dtype = torch.int64)
        IM[V] = I

        return (IM[E], Xe, Xv, labels)

    def __len__(self):
        return len(self.starts)

class DataLoader:
    def __init__(self, paths, sequenceLength, sequenceStride = None, edgeFeatures = set(), nodeFeatures = set(), align = 1):
        self.paths = paths
        self.sequenceLength = sequenceLength
        self.sequenceStride = sequenceStride or sequenceLength
        self.edgeFeatures = edgeFeatures
        self.nodeFeatures = nodeFeatures
        self.align = align

    def loadData(self, tasks):
        columns = set(['ip_src', 'ip_dst', 'timestamp'])
        columns.update(self.edgeFeatures)
        columns.update(*(task.required_features for task in tasks))

        files = tqdm([file for path in self.paths for file in glob.iglob(path)], desc = 'loading data')
        data = pd.concat((pd.read_csv(file, usecols = columns) for file in files), ignore_index = True, copy = False)

        with TimedStep('Sorting'):
            data.sort_values('timestamp', kind = 'quicksort', inplace = True, ignore_index = True)

        with TimedStep('Converting node IPs'):
            id2ip = pd.unique(data[['ip_src', 'ip_dst']].values.ravel('K'))
            ip2id = {ip: id for id, ip in enumerate(id2ip)}
            data.loc[:,('ip_src', 'ip_dst')] = data[['ip_src', 'ip_dst']].applymap(ip2id.get)
            E = pandas2torch(data[['ip_src', 'ip_dst']], dtype = np.int64)

        with TimedStep('Computing labels'):
            num_outputs = max(task.output for task in tasks) + 1
            labels = tuple([] for _ in range(num_outputs))
            for task in tasks:
                labels[task.output].append(task.compute_labels(data, id2ip, ip2id))
            labels = tuple(torch.stack(lbls, dim = -1) for lbls in labels)

        with TimedStep('Computing node features'):
            if len(self.nodeFeatures) > 0:
                ips = [ipaddress.IPv4Address(ip) for ip in id2ip]
                features = []
                if 'is-private' in self.nodeFeatures:
                    isPrivate = numpy2torch([ip.is_private for ip in ips])
                    features.append(isPrivate)
                if 'is-multicast' in self.nodeFeatures:
                    isMulticast = numpy2torch([ip.is_multicast for ip in ips])
                    features.append(isMulticast)
                Xv = torch.stack(features, dim = 1)
            else:
                Xv = torch.ones(len(id2ip), 1)

        with TimedStep('Computing edge features'):
            Xnum = data[self.edgeFeatures].select_dtypes(include = np.number)
            q1, med, q3 = map(lambda t: t[1], Xnum.quantile([.25,.5,.75]).iterrows())
            Xnum = (Xnum - med) / (q3 - q1)

            Xcat = data[self.edgeFeatures].select_dtypes(exclude = np.number)
            Xcat = pd.get_dummies(Xcat)

            Xe = Xnum.join(Xcat)

            totalFeatures = 2 * Xv.shape[1] + Xe.shape[1]
            if totalFeatures % self.align != 0:
                for i in range(self.align - (totalFeatures % self.align)):
                    Xe[f'_pad_{i}'] = 0

            Xe = pandas2torch(Xe)

        numSequences = int((E.shape[0] - self.sequenceLength) / self.sequenceStride)
        starts = np.arange(numSequences) * self.sequenceStride
        dataset = SequenceDataset((E, Xe, labels), self.sequenceLength, starts, Xv)

        info = {
            'N': Xv.shape[0],
            'E': Xe.shape[0],
            'Fv': Xv.shape[1],
            'Fe': Xe.shape[1],
            'Nseqs': len(dataset),
            'id2ip': list(id2ip),
            'ip2id': ip2id
        }

        return dataset, info