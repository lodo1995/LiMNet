import argparse
import pandas as pd
import torch
import util

def load_state_dict(baseFolder, fromCheckpoint):
    if fromCheckpoint:
        state_dict = torch.load(util.latestCheckpoint(baseFolder), map_location = 'cpu')['model']
    else:
        state_dict = torch.load(f'{baseFolder}/model.pt', map_location = 'cpu')
    return state_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Print size of trained model.')
    parser.add_argument('run_folder', type = str, help = 'the trained output must be saved in this folder')
    parser.add_argument('--from-checkpoint', action = 'store_true', help = 'use the latest checkpoint file instead of the model.pt file')
    args = parser.parse_args()
    
    baseFolder = f'{args.run_folder}'

    state_dict = load_state_dict(baseFolder, args.from_checkpoint)

    entries = []
    total = 0
    for key, tensor in state_dict.items():
        size = tensor.nelement() * 4
        total += size
        entries.append((key, tuple(tensor.shape), size / 1024))
    entries.append(('TOTAL', '', total / 1024))

    weights = pd.DataFrame(entries, columns = ['Tensor', 'Shape', 'Size [KiB]'])
    weights['Category'] = weights.Tensor.str.slice(0,5)
    weights = weights.groupby('Category', as_index = False).sum()

    print(weights)