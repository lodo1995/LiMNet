import torch
from tqdm.auto import tqdm
from typing import Tuple, Optional

class LSTMMemoryUpdater(torch.nn.Module):
    def __init__(self, Fh, cellType = torch.nn.LSTMCell):
        super().__init__()
        self.Fh = Fh
        self.cellType = cellType
        
    def build(self, Fe, Fv):
        Fi = self.Fh + 2*Fv + Fe
        self.incoming = self.cellType(Fi, self.Fh)
        self.outgoing = self.cellType(Fi, self.Fh)
        return self.Fh

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], states: Tuple[torch.Tensor, torch.Tensor]):
        E, Xe, Xv = inputs
        H, C = states
        B = torch.arange(H.shape[0])

        src, dst = E[:,0], E[:,1]
        srcStates = H[B,src], C[B,src]
        dstStates = H[B,dst], C[B,dst]
        srcFeatures = Xv[:,0,:]
        dstFeatures = Xv[:,1,:]
        srcInputs = torch.hstack((dstStates[0], srcFeatures, dstFeatures, Xe))
        dstInputs = torch.hstack((srcStates[0], srcFeatures, dstFeatures, Xe))

        Hsrc, C[B,src] = self.outgoing(srcInputs, srcStates)
        Hdst, C[B,dst] = self.incoming(dstInputs, dstStates)
        H[B,src], H[B,dst] = Hsrc, Hdst
        return (Hsrc, Hdst), (H, C)

    @torch.jit.export
    def init_memory(self, B: int, N: int, device: torch.device, dtype: Optional[torch.dtype] = None):
        H = torch.zeros(B, N, self.Fh, device = device, dtype = dtype)
        C = torch.zeros(B, N, self.Fh, device = device, dtype = dtype)
        return (H, C)

class SimpleMemoryUpdater(torch.nn.Module):
    def __init__(self, Fh, cellType):
        super().__init__()
        self.Fh = Fh
        self.cellType = cellType
        
    def build(self, Fe, Fv):
        Fi = self.Fh + 2*Fv + Fe
        self.incoming = self.cellType(Fi, self.Fh)
        self.outgoing = self.cellType(Fi, self.Fh)
        return self.Fh

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], H: torch.Tensor):
        E, Xe, Xv = inputs
        B = torch.arange(H.shape[0])

        src, dst = E[:,0], E[:,1]
        Hsrc, Hdst = H[B,src], H[B,dst]
        srcFeatures = Xv[:,0,:]
        dstFeatures = Xv[:,1,:]
        srcInputs = torch.hstack((Hdst, srcFeatures, dstFeatures, Xe))
        dstInputs = torch.hstack((Hsrc, srcFeatures, dstFeatures, Xe))

        Hsrc = self.outgoing(srcInputs, Hsrc)
        Hdst = self.incoming(dstInputs, Hdst)
        H[B,src], H[B,dst] = Hsrc, Hdst
        return (Hsrc, Hdst), H

    @torch.jit.export
    def init_memory(self, B: int, N: int, device: torch.device, dtype: Optional[torch.dtype] = None):
        H = torch.zeros(B, N, self.Fh, device = device, dtype = dtype)
        return H

class EdgeDecoder(torch.nn.Module):
    def __init__(self, H):
        super().__init__()
        self.H = H

    def build(self, Fh, Fe, Fv):
        self.linear = torch.nn.Linear(2*Fh + Fe, self.H)

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], embeddings: Tuple[torch.Tensor, torch.Tensor]):
        E, Xe, Xv = inputs
        Hsrc, Hdst = embeddings
        B = torch.arange(Hsrc.shape[0])
        emb = torch.hstack((Hsrc, Xe, Hdst))
        return self.linear(emb)

class NodeDecoder(torch.nn.Module):
    def __init__(self, H):
        super().__init__()
        self.H = H

    def build(self, Fh, Fe, Fv):
        self.linear = torch.nn.Linear(Fh, self.H)

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], embeddings: Tuple[torch.Tensor, torch.Tensor]):
        E, _, Xv = inputs
        Hsrc, Hdst = embeddings
        B = torch.arange(Hsrc.shape[0])
        emb = torch.stack((Hsrc, Hdst), dim = 1)
        return self.linear(emb)

class MemoryNetwork(torch.nn.Module):
    def __init__(self, memory_updater):
        super().__init__()
        self.memory_updater = memory_updater

    def build(self, info, decoders):
        Fh = self.memory_updater.build(info['Fe'], info['Fv'])
        self.decoders = decoders
        for i, decoder in enumerate(self.decoders):
            decoder.build(Fh, info['Fe'], info['Fv'])
            self.add_module(f'decoder_{i}', decoder)

    def optimize(self):
        self.memory_updater = torch.jit.script(self.memory_updater)
        self.decoders = tuple(torch.jit.script(decoder) for decoder in self.decoders)

    def forward(self, inputs, memory = None):
        E, Xe, Xv = inputs
        T, B, _ = E.shape
        N = torch.max(E) + 1

        if memory is None:
            memory = self.memory_updater.init_memory(B, N, Xv.device)

        outputs = tuple([] for _ in range(len(self.decoders)))

        for inputs in tqdm(zip(E, Xe, Xv), total = float(T), desc = 'sequence', leave = False, mininterval = 0.5):
            embs, memory = self.memory_updater(inputs, memory)
            for outs, decoder in zip(outputs, self.decoders):
                outs.append(decoder(inputs, embs))

        return tuple(torch.stack(outs) for outs in outputs)

    @staticmethod
    def makeBatch(seqs):
        E  = torch.stack([E  for E, Xe, Xv, labels in seqs], dim = 1)
        Xe = torch.stack([Xe for E, Xe, Xv, labels in seqs], dim = 1)
        Xv = torch.stack([Xv for E, Xe, Xv, labels in seqs], dim = 1)
        labels = tuple(torch.stack([labels[i] for E, Xe, Xv, labels in seqs], dim = 1) for i in range(len(seqs[0][3])))
        return (E, Xe, Xv), labels