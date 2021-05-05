import torch
from typing import Tuple

class FastGRNN(torch.nn.Module):
    def __init__(self, Fi, Fh):
        super().__init__()
        self.input_linear = torch.nn.Linear(Fi, Fh, bias = False)
        self.state_linear = torch.nn.Linear(Fh, Fh, bias = False)
        self.gate_bias = torch.nn.Parameter(torch.ones(1, Fh))
        self.update_bias = torch.nn.Parameter(torch.ones(1, Fh))
        self.zeta = torch.nn.Parameter(torch.ones(1, 1))
        self.nu = torch.nn.Parameter(-4 * torch.ones(1, 1))

    def forward(self, input, state):
        merged = self.input_linear(input) + self.state_linear(state)
        gate = torch.sigmoid(merged + self.gate_bias)
        update = torch.tanh(merged + self.update_bias)
        delta = torch.sigmoid(self.zeta) * (1 - gate) + torch.sigmoid(self.nu)
        return gate * state + delta * update

class LSTM(torch.nn.Module):
    def __init__(self, Fi, Fh):
        super().__init__()
        self.input_linear = torch.nn.Linear(Fi, 4 * Fh)
        self.state_linear = torch.nn.Linear(Fh, 4 * Fh, bias = False)

    def forward(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor]):
        H, C = states

        gates = self.input_linear(inputs) + self.state_linear(H)
        I, F, O, G = gates.chunk(4, 1)

        I = torch.sigmoid(I)
        F = torch.sigmoid(F)
        O = torch.sigmoid(O)
        G = torch.tanh(G)

        C = F * C + I * G
        H = O * torch.tanh(C)

        return H, C

class GRU(torch.nn.Module):
    def __init__(self, Fi, Fh):
        super().__init__()
        self.input_linear = torch.nn.Linear(Fi, 3 * Fh)
        self.state_linear = torch.nn.Linear(Fh, 3 * Fh)

    def forward(self, X: torch.Tensor, H: torch.Tensor):
        Xr, Xz, Xn = self.input_linear(X).chunk(3, 1)
        Hr, Hz, Hn = self.state_linear(H).chunk(3, 1)

        R = torch.sigmoid(Xr + Hr)
        Z = torch.sigmoid(Xz + Hz)
        N = torch.tanh(Xn + R * Hn)

        H = (1 - Z) * N + Z * H

        return H