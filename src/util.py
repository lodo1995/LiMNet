import glob
import importlib
import numpy as np
import os
import re
import sys
import time
import torch
from ignite.engine import Events
from tqdm.auto import tqdm

class TimedStep:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        tqdm.write(f'{self.name}...')
        self.start_time = time.perf_counter()

    def __exit__(self, *exc):
        t = time.perf_counter()
        tqdm.write('  Done {:.2f} s'.format(t - self.start_time))

def numpy2torch(data, dtype = np.float32):
    return torch.from_numpy(dtype(data))

def pandas2torch(data, dtype = np.float32):
    return numpy2torch(data.to_numpy(), dtype)

def loadConfigurationFile(configFile):
    confDir, confModule = os.path.split(configFile)
    if confDir != '':
        sys.path.append(confDir)
    if confModule.endswith('.py'):
        confModule = confModule[:-3]
    return importlib.import_module(confModule)

def latestCheckpoint(baseFolder):
    extractLastNumber = lambda s: int(re.findall('\d+', s)[-1])
    return sorted(glob.glob(f'{baseFolder}/checkpoint*'), key = extractLastNumber)[-1]

def tqdmHandler(engine, epochs = False, **kwargs):
    startEvent = Events.STARTED if epochs else Events.EPOCH_STARTED
    stepEvent = Events.EPOCH_COMPLETED if epochs else Events.ITERATION_COMPLETED
    closeEvent = Events.COMPLETED if epochs else Events.EPOCH_COMPLETED

    pbar = None

    @engine.on(startEvent)
    def create_pbar(engine):
        nonlocal pbar
        total = engine.state.max_epochs if epochs else engine.state.epoch_length
        initial = engine.state.epoch if epochs else 0
        pbar = tqdm(total = total, initial = initial, **kwargs)
    
    @engine.on(stepEvent)
    def update_pbar(engine):
        pbar.update(1)

    @engine.on(closeEvent)
    def close_pbar(engine):
        nonlocal pbar
        pbar.close()
        pbar = None

def split_dataset(dataset, baseFolder, resume, trainRatio):
    if resume:
        with open(f'{baseFolder}/seed.txt', 'r') as f:
            seed = int(f.read())
    else:        
        seed = np.random.randint(2**31)
        with open(f'{baseFolder}/seed.txt', 'w') as f:
            f.write(str(seed))
    torch.manual_seed(seed)

    trainCount = int(len(dataset) * trainRatio)
    validCount = len(dataset) - trainCount
    return torch.utils.data.random_split(dataset, [trainCount, validCount])