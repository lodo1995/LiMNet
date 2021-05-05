import cells
from network_sequence_loader import DataLoader as NetworkDataLoader
from packet_header_loader    import DataLoader as PacketDataLoader
from metrics import accuracy, auc
import memory_model as mem
import language_model as lang
import outputs as o
import torch

trials = 1

datasets = {
    'kitsune':  ('kitsune/*'        , 1024, 30,   8, 20),
    'torii':    ('medbiot/torii*'   , 4096, 10,  32, 10),
    'mirai':    ('medbiot/mirai*'   , 4096, 10,  64, 10),
    'bashlite': ('medbiot/bashlite*', 8192, 10, 192, 10),
    'medbiot':  ('medbiot/*'        , 8192, 10, 192, 10),
}

cells = {
    'fastgrnn': (cells.FastGRNN   , lang.SimpleRecurrentLayer, False, mem.SimpleMemoryUpdater, False),
    'gru':      (torch.nn.GRUCell , lang.SimpleRecurrentLayer, True , mem.SimpleMemoryUpdater, False),
    'lstm':     (torch.nn.LSTMCell, lang.LSTMRecurrentLayer  , True , mem.LSTMMemoryUpdater  , False),
}

recurrentLayers = [1, 3]

featureSets = {
    'all_feats':      PacketDataLoader.ALL_FEATURES           ,
    'no_ids':         PacketDataLoader.ALL_FEATURES_EXCEPT_IDS,
    'ports_and_size': PacketDataLoader.PORT_AND_SIZE_ONLY     ,
}

embeddingSizes = [32, 64]

configs = {}

for datasetName, (datasetLocation, recTrainBatchSize, recEpochs, memTrainBatchSize, memEpochs) in datasets.items():
    for cellName, (cellType, recLayerType, recMixedPrecision, memLayerType, memMixedPrecision) in cells.items():
        for embeddingSize in embeddingSizes:

            for numLayers in recurrentLayers:
                for featureSetName, featureSet in featureSets.items():
                    configName = f'{datasetName}-lang-{cellName}-{numLayers}-{embeddingSize}-{featureSetName}'
                    configs[configName] = {
                        'dataLoader': PacketDataLoader(    
                            paths = [f'data/headers/{datasetLocation}'],
                            edgeFeatures = featureSet
                        ),
                        'trainRatio': 0.8,
                        'epochs': recEpochs,
                        'trainBatchSize': recTrainBatchSize,
                        'validBatchSize': recTrainBatchSize,
                        'model': lang.RecurrentNetwork(embeddingSize, [recLayerType(cellType, embeddingSize, 0.2) for _ in range(numLayers)]),
                        'optimizer': torch.optim.Adam,
                        'mixedPrecision': recMixedPrecision,
                        'outputs': [
                            (lang.NodeDecoder, [
                                o.NodeIsMalicious(metrics = [accuracy, auc]),
                                o.NodeIsAttacked(metrics = [accuracy, auc]),
                            ]),
                            (lang.EdgeDecoder, [
                                o.EdgeIsMalicious(metrics = [accuracy, auc]),
                            ]),
                        ]
                    }

            configName = f'{datasetName}-mem-{cellName}-{embeddingSize}'
            configs[configName] = {
                'dataLoader': NetworkDataLoader(
                    paths = [f'data/{datasetLocation}'],
                    edgeFeatures = {'length', 'protocol'},
                    nodeFeatures = {'is-private', 'is-multicast'},
                    sequenceLength = 5_000,
                    sequenceStride = 1_000,
                ),
                'trainRatio': 0.8,
                'epochs': memEpochs,
                'trainBatchSize': memTrainBatchSize,
                'validBatchSize': 2048,
                'model': mem.MemoryNetwork(memLayerType(embeddingSize, cellType)),
                'optimizer': torch.optim.Adam,
                'mixedPrecision': memMixedPrecision,
                'outputs': [
                    (mem.NodeDecoder, [
                        o.NodeIsMalicious(metrics = [accuracy, auc]),
                        o.NodeIsAttacked(metrics = [accuracy, auc]),
                    ]),
                    (mem.EdgeDecoder, [
                        o.EdgeIsMalicious(metrics = [accuracy, auc]),
                    ]),
                ]
            }