import glob
import json
import numpy as np
import pandas as pd
from scipy.stats.mstats import gmean

metric_names = [
    'node-is-malicious-accuracy', 'node-is-malicious-auc',
    'node-is-attacked-accuracy', 'node-is-attacked-auc',
    'edge-is-malicious-accuracy', 'edge-is-malicious-auc'
]

def read_results(path):
    try:
        dataset, _, cell, layers, emb_size, feats = os.path.basename(path).split('-')
        kind = 'recurrent'
    except:
        dataset, _, cell, emb_size = os.path.basename(path).split('-')
        layers = '1'
        feats = 'ports_and_size'
        kind = 'memory'
    metrics = []
    for trial_path in glob.iglob(f'{path}/trial-*'):
        if os.path.exists(f'{trial_path}/metrics.json'):
            with open(f'{trial_path}/metrics.json', 'r') as f:
                metrics.append(json.load(f))
    output = {
        'kind': kind,
        'dataset': dataset,
        'num_layers': layers,
        'emb_size': emb_size,
        'feats': feats,
        'cell_type': cell,
    }
    for metric_name in metric_names:
        output[metric_name] = np.mean([max(epoch_metrics[metric_name] for epoch_metrics in trial_metrics) for trial_metrics in metrics])
    return output

def read_all(pattern):
    for path in glob.iglob(pattern):
        if os.path.isdir(path) and not '__pycache__' in path:
            yield read_results(path)
            
def to_latex(val, max):
    if val == max:
        template = '\\textbf{', '}'
    elif val >= max * (1 - 0.005):
        template = '\\underline{', '}'
    else:
        template = '', ''
    val = round(val * 100, 2)
    return template[0] + str(val) + template[1]

def latex_prepare(results):
    params = ['kind', 'num_layers', 'emb_size', 'cell_type']
    aucs = metric_names[1::2]
    results['geom_mean'] = gmean(results[aucs], axis = 1)
    metrics = aucs + ['geom_mean']
    maxes = {metr: results[metr].max() for metr in metrics}
    results.sort_values(params, inplace = True)
    for _, entry in results.iterrows():
        data = [entry[col] for col in params]
        data.extend(to_latex(entry[metr], maxes[metr]) for metr in metrics)
        print(f'{" & ".join(data)} \\\\ \\hline')

if __name__ == '__main__':
    # Table 3
    # results = pd.DataFrame.from_records(list(read_all('multiruns/all-configs/kitsune-lang*')))
    # best_by_feats = results.groupby('feats').max()
    # best_by_feats.kind = best_by_feats.index
    # results = results[results.feats == 'ports_and_size']
    # results = results.append(best_by_feats.loc['all_feats'])
    # results = results.append(best_by_feats.loc['no_ids'])
    # latex_prepare(results)

    # Table 4
    results = pd.DataFrame.from_records(list(read_all('multiruns/all-configs/kitsune-mem*')))
    results = results[results.feats == 'ports_and_size']
    results = results.append(pd.DataFrame.from_records(list(read_all('multiruns/all-configs/kitsune-lang-lstm-1-64-p*'))))
    results = results.append(pd.DataFrame.from_records(list(read_all('multiruns/all-configs/kitsune-lang-fastgrnn-3-64-p*'))))
    results = results.append(pd.DataFrame.from_records(list(read_all('multiruns/all-configs/kitsune-lang-gru-3-64-p*'))))
    latex_prepare(results)