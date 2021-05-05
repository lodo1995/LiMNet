import ignite
import ignite.contrib.metrics
import torch

def accuracy(prepare_predictions, device):
    def output_transform(state_output):
        ypred, ytrue = prepare_predictions(state_output)
        return torch.round(torch.squeeze(ypred)), torch.squeeze(ytrue)
    return ignite.metrics.Accuracy(output_transform, device = device)

def auc(prepare_predictions, device):
    def output_transform(state_output):
        ypred, ytrue = prepare_predictions(state_output)
        return torch.flatten(ypred), torch.flatten(ytrue)
    return ignite.contrib.metrics.ROC_AUC(output_transform)

def fbeta(beta):
    def _fbeta(prepare_predictions, device):
        def output_transform(state_output):
            ypred, ytrue = prepare_predictions(state_output)
            return torch.round(torch.squeeze(ypred)), torch.squeeze(ytrue)
        return ignite.metrics.Fbeta(beta, output_transform = output_transform, device = device)
    return _fbeta

f1 = fbeta(1)