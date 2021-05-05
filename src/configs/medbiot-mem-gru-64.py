import cells
from network_sequence_loader import DataLoader
from metrics import accuracy, auc
import memory_model as m
import outputs as o
import torch

conf = {
    'dataLoader'    : DataLoader(
        paths          = ['data/medbiot/*'],
        edgeFeatures   = {'length', 'protocol'},
        nodeFeatures   = {'is-private', 'is-multicast'},
        sequenceLength = 10_000,
        sequenceStride = 1_000,
    ),
    'trainRatio'    : 0.8,
    'epochs'        : 5,
    'trainBatchSize': 192,
    'validBatchSize': 4096,
    'model'         : m.MemoryNetwork(m.SimpleMemoryUpdater(64, torch.nn.GRUCell)),
    'optimizer'     : torch.optim.Adam,
    'mixedPrecision': True,
    'outputs'       : [
        (m.NodeDecoder, [
            o.NodeIsMalicious(metrics = [accuracy, auc]),
            o.NodeIsAttacked (metrics = [accuracy, auc]),
        ]),
        (m.EdgeDecoder, [
            o.EdgeIsMalicious(metrics = [accuracy, auc]),
        ]),
    ]
}