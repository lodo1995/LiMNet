import torch
from typing import Optional, Tuple

class SimpleRecurrentLayer(torch.nn.Module):
    def __init__(self, cell_type, size, dropout_rate):
        super().__init__()
        self.cell_type = cell_type
        self.size = size
        self.dropout_rate = dropout_rate

    def build(self, size_in):
        self.cell = self.cell_type(size_in, self.size)
        self.dropout = torch.nn.Dropout(self.dropout_rate)
        return self.size

    def forward(self, inputs: torch.Tensor):
        H = torch.zeros((inputs.shape[1], self.size), dtype = inputs.dtype, device = inputs.device)
        outputs = []
        for input in inputs:
            H = self.cell(input, H)
            outputs.append(H)
        return torch.stack(outputs)

class LSTMRecurrentLayer(torch.nn.Module):
    def __init__(self, cell_type, size, dropout_rate):
        super().__init__()
        self.cell_type = cell_type
        self.size = size
        self.dropout_rate = dropout_rate

    def build(self, size_in):
        self.cell = self.cell_type(size_in, self.size)
        self.dropout = torch.nn.Dropout(self.dropout_rate)
        return self.size

    def forward(self, inputs: torch.Tensor):
        H = torch.zeros((inputs.shape[1], self.size), dtype = inputs.dtype, device = inputs.device)
        C = torch.zeros((inputs.shape[1], self.size), dtype = inputs.dtype, device = inputs.device)
        outputs = []
        for input in inputs:
            H, C = self.cell(input, (H, C))
            outputs.append(H)
        return torch.stack(outputs)

class EdgeDecoder(torch.nn.Module):
    def __init__(self, H):
        super().__init__()
        self.H = H

    def build(self, Fh):
        self.linear = torch.nn.Linear(Fh, self.H)

    def forward(self, input: torch.Tensor):
        return self.linear(input)

class NodeDecoder(torch.nn.Module):
    def __init__(self, H):
        super().__init__()
        self.H = H

    def build(self, Fh):
        self.src_linear = torch.nn.Linear(Fh, self.H)
        self.dst_linear = torch.nn.Linear(Fh, self.H)

    def forward(self, input: torch.Tensor):
        src_out = self.src_linear(input)
        dst_out = self.dst_linear(input)
        return torch.stack((src_out, dst_out), dim = 1)

class RecurrentNetwork(torch.nn.Module):
    def __init__(self, emb_dim, layers):
        super().__init__()
        self.emb_dim = emb_dim
        self.layers = tuple(layers)

    def build(self, info, decoders):
        self.embedder = torch.nn.Embedding(info['Ntoks'], self.emb_dim)
        Fh = self.emb_dim
        for i, layer in enumerate(self.layers):
            Fh = layer.build(Fh)
            self.add_module(f'layer_{i}', layer)
        self.decoders = decoders
        for i, decoder in enumerate(self.decoders):
            decoder.build(Fh)
            self.add_module(f'decoder_{i}', decoder)

    def optimize(self):
        self.embedder = torch.jit.script(self.embedder)
        self.layers = tuple(torch.jit.script(layer) for layer in self.layers)
        self.decoders = tuple(torch.jit.script(decoder) for decoder in self.decoders)

    def forward(self, input):
        outputs = tuple([] for _ in range(len(self.decoders)))

        embs = self.embedder(input)
        for layer in self.layers:
            embs = layer(embs)
        for outs, decoder in zip(outputs, self.decoders):
            outs.append(decoder(embs[-1]))

        return tuple(torch.stack(outs) for outs in outputs)

    @staticmethod
    def makeBatch(seqs):
        Xe = torch.stack([Xe for Xe, labels in seqs], dim = 1)
        labels = tuple(torch.stack([labels[i] for Xe, labels in seqs], dim = 0) for i in range(len(seqs[0][1])))
        return Xe, labels