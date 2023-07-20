import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from utils import make_spectrogram


class SpectrogramENZ(nn.Module):
    def __init__(self):
        super(SpectrogramENZ, self).__init__()

    def forward(self, x):
        E, N, Z = x[:, 0], x[:, 1], x[:, 2]
        wave = torch.cat([E, N, Z], dim=1).cpu()
        output = Tensor(make_spectrogram(wave=wave)).cuda()
        return output


class ClassificationDecoder(nn.Module):
    def __init__(self, dim=160, num_classes=2):
        super(ClassificationDecoder, self).__init__()
        self.fc1 = nn.Linear(dim, 32)
        self.fc2 = nn.Linear(32, num_classes)
        self.fc3 = nn.Linear(39, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).permute(0, 2, 1)
        x = F.relu(x)
        x = self.fc3(x)
        return x


class MagnitudeEstimationDecoder(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 bias: bool = True,
                 batch_first: bool = True,
                 bidirectional: bool = True):
        super(MagnitudeEstimationDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            bias=bias,
                            batch_first=batch_first,
                            bidirectional=bidirectional)
        self.fc = nn.Linear(in_features=input_size * 2,
                            out_features=1)

    def forward(self, x):
        output, (_, _) = self.lstm(x)
        output = output[:, -1]
        output = self.fc(output).squeeze(0)
        return output
