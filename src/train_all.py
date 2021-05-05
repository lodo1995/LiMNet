import argparse
import copy
import os
import util
import torch
from training import run_training

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Train multiple model configurations.')
    parser.add_argument('root_folder', type = str, help = 'the results for each configuration will be stored in folder multiruns/<root_folder>/<config_name>')
    parser.add_argument('configs', type = str, nargs = '?', help = 'path of the file containing all the configurations to train')
    parser.add_argument('--gpu', type = int, nargs = '?', const = 0, help = 'whether to use a GPU; a 0-based index can optionally specify which GPU to use')
    args = parser.parse_args()

    rootFolder = f'multiruns/{args.root_folder}'
    config = util.loadConfigurationFile(args.configs)
    device = torch.device(f'cuda:{args.gpu}' if args.gpu is not None else 'cpu')

    if not os.path.exists(rootFolder):
        os.makedirs(rootFolder)

    with open(f'{rootFolder}/configs.py', 'w') as target, \
         open(args.configs, 'r') as input:
        target.write(input.read())

    i = 1
    total = len(config.configs) * config.trials

    for name, conf in config.configs.items():
        for t in range(config.trials):

            step = f'({i}/{total})'
            i += 1

            baseFolder = f'{rootFolder}/{name}/trial-{t}'

            if os.path.exists(baseFolder):
                print(f'{step} Skipping already existing {baseFolder}')
            else:
                print(f'{step} Running {baseFolder}')
                os.makedirs(baseFolder)
                run_training(copy.deepcopy(conf), baseFolder, device, False)