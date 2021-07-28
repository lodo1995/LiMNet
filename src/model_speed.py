import argparse
import ignite
import os
import outputs as o
import torch
import util
from ignite.engine import Events

def measure_speed(baseFolder, conf, samples):
    model = conf['model']
    outputs = conf['outputs']
    dataLoader = conf['dataLoader']
    tasks = o.get_tasks(outputs, torch.device('cpu'))

    dataset, info = dataLoader.loadData(tasks)
    _, validDataset = util.split_dataset(dataset, baseFolder, True, conf['trainRatio'])
    validData = torch.utils.data.DataLoader(validDataset, batch_size = 1, collate_fn = model.makeBatch, num_workers = 2)

    torch.set_num_threads(1)
    model.build(info, o.get_decoders(outputs))

    try:
        model.load_state_dict(torch.load(f'{baseFolder}/model.pt', map_location = 'cpu'))
    except:
        model.load_state_dict(torch.load(util.latestCheckpoint(baseFolder), map_location = 'cpu')['model'])
        
    model.optimize()
    metrics = {metricKey: metricFunc for task in tasks for metricKey, metricFunc in task.metrics.items()}

    evaluator = ignite.engine.create_supervised_evaluator(model)

    timer = ignite.handlers.Timer(average=True)
    timer.attach(evaluator, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                            pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    util.tqdmHandler(evaluator)
    @evaluator.on(ignite.engine.Events.ITERATION_COMPLETED(once = samples))
    def terminate_evaluation(evaluator):
        evaluator.terminate_epoch()

    evaluator.run(validData)

    print(f'Average packets/s: {dataLoader.sequenceLength / timer.value()}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Measure speed of a trained model.')
    parser.add_argument('base_folder', type = str, help = 'the folder containing the trained model')
    parser.add_argument('--samples', type = int, default = 10, help = 'number of samples to run (higher means more precise speed results')
    args = parser.parse_args()
    
    try:
        baseFolder = f'runs/{args.run_name}'
        confFile = f'{baseFolder}/conf.py'
        conf = util.loadConfigurationFile(confFile).conf
    except ModuleNotFoundError:
        baseFolder = os.path.normpath(args.run_name)
        confFile = os.path.normpath(f'{baseFolder}/../../configs.py')
        cfg = util.loadConfigurationFile(confFile)
        conf = cfg.configs[baseFolder.split('/')[-2]]

    measure_speed(baseFolder, conf, args.samples)