import torch
import torch.nn as nn
from modules import SpectrogramENZ
from utils import make_spectrogram
from conformer.attention import MultiHeadedSelfAttentionModule


class EQTransformerEncoder(nn.Module):
    def __init__(self):
        super(EQTransformerEncoder, self).__init__()
        self.conv1d_1 = nn.Conv1d(3, 8, 1, 1)
        self.max_pooling1d_1 = nn.MaxPool1d(2)
        self.conv1d_2 = nn.Conv1d(8, 16, 1, 1)
        self.max_pooling1d_2 = nn.MaxPool1d(2)
        self.conv1d_3 = nn.Conv1d(16, 16, 1, 1)
        self.max_pooling1d_3 = nn.MaxPool1d(2)
        self.conv1d_4 = nn.Conv1d(16, 32, 1, 1)
        self.max_pooling1d_4 = nn.MaxPool1d(2)
        self.conv1d_5 = nn.Conv1d(32, 32, 1, 1)
        self.max_pooling1d_5 = nn.MaxPool1d(2)
        self.conv1d_6 = nn.Conv1d(32, 64, 1, 1)
        self.max_pooling1d_6 = nn.MaxPool1d(2)
        self.conv1d_7 = nn.Conv1d(64, 64, 1, 1)
        self.max_pooling1d_7 = nn.MaxPool1d(2)

        self.batch_normalization_1 = nn.BatchNorm1d(64)
        self.activation_1 = nn.ReLU()

        self.conv1d_8 = nn.Conv1d(64, 64, 1, 1)
        self.batch_normalization_2 = nn.BatchNorm1d(64)
        self.activation_2 = nn.ReLU()

        self.conv1d_9 = nn.Conv1d(64, 64, 1, 1)
        self.batch_normalization_3 = nn.BatchNorm1d(64)
        self.activation_3 = nn.ReLU()

        self.conv1d_10 = nn.Conv1d(64, 64, 1, 1)
        self.batch_normalization_4 = nn.BatchNorm1d(64)
        self.activation_4 = nn.ReLU()

        self.conv1d_11 = nn.Conv1d(64, 64, 1, 1)
        self.bidirectional_1 = nn.LSTM(64, 32)

        self.conv1d_12 = nn.Conv1d(32, 16, 1, 1)
        self.batch_normalization_5 = nn.BatchNorm1d(16)
        self.attentionD0 = MultiHeadedSelfAttentionModule(d_model=16,
                                                          num_heads=1,
                                                          dropout_p=0.1)
        self.layer_normalization_1 = nn.LayerNorm(16)
        self.feed_forward_1 = nn.Linear(16, 16)

        self.layer_normalization_2 = nn.LayerNorm(16)
        self.attentionD = MultiHeadedSelfAttentionModule(d_model=16,
                                                         num_heads=1,
                                                         dropout_p=0.1)

        self.layer_normalization_3 = nn.LayerNorm(16)
        self.feed_forward_2 = nn.Linear(16, 16)

        self.layer_normalization_4 = nn.LayerNorm(16)

        self.lstm_2 = nn.LSTM(16, 16)
        self.lstm_3 = nn.LSTM(16, 16)

    def forward(self, x):
        x = self.max_pooling1d_1(self.conv1d_1(x))
        x = self.max_pooling1d_2(self.conv1d_2(x))
        x = self.max_pooling1d_3(self.conv1d_3(x))
        x = self.max_pooling1d_4(self.conv1d_4(x))
        x = self.max_pooling1d_5(self.conv1d_5(x))
        x = self.max_pooling1d_6(self.conv1d_6(x))
        x = self.max_pooling1d_7(self.conv1d_7(x))

        res = x
        x = self.activation_1(self.batch_normalization_1(x))
        x = self.conv1d_8(x)
        x = self.activation_2(self.batch_normalization_2(x))
        x = self.conv1d_9(x) + res

        res = x
        x = self.activation_3(self.batch_normalization_3(x))
        x = self.conv1d_10(x)
        x = self.activation_4(self.batch_normalization_4(x))
        x = self.conv1d_11(x) + res

        x = x.permute(0, 2, 1)
        x, (_, _) = self.bidirectional_1(x)
        x = x.permute(0, 2, 1)
        x = self.conv1d_12(x)
        x = self.batch_normalization_5(x)

        res = x.permute(0, 2, 1)
        x = x.permute(0, 2, 1)
        x = self.attentionD0(x) + res

        x = self.layer_normalization_1(x)

        res = x
        x = self.feed_forward_1(x) + res

        x = self.layer_normalization_2(x)
        res = x
        x = self.attentionD(x) + res

        x = self.layer_normalization_3(x)
        res = x
        x = self.feed_forward_2(x) + res

        x = self.layer_normalization_4(x)
        x, (_, _) = self.lstm_2(x)
        x, (_, _) = self.lstm_3(x)
        return x


class EQTransformerDecoder(nn.Module):
    def __init__(self):
        super(EQTransformerDecoder, self).__init__()
        self.up_sampling1d_1 = nn.Upsample(94)
        self.conv1d_1 = nn.Conv1d(16, 64, 1, 1)
        self.up_sampling1d_2 = nn.Upsample(188)
        self.conv1d_2 = nn.Conv1d(64, 64, 1, 1)
        self.up_sampling1d_3 = nn.Upsample(376)
        self.conv1d_3 = nn.Conv1d(64, 32, 1, 1)
        self.up_sampling1d_4 = nn.Upsample(752)
        self.conv1d_4 = nn.Conv1d(32, 32, 1, 1)
        self.up_sampling1d_5 = nn.Upsample(1500)
        self.conv1d_5 = nn.Conv1d(32, 16, 1, 1)
        self.up_sampling1d_6 = nn.Upsample(3000)
        self.conv1d_6 = nn.Conv1d(16, 16, 1, 1)
        self.up_sampling1d_6 = nn.Upsample(6000)
        self.conv1d_7 = nn.Conv1d(16, 8, 1, 1)
        self.conv1d_8 = nn.Conv1d(8, 1, 1, 1)
        self.fc = nn.Linear(6000, 2)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1d_1(self.up_sampling1d_1(x))
        x = self.conv1d_2(self.up_sampling1d_2(x))
        x = self.conv1d_3(self.up_sampling1d_3(x))
        x = self.conv1d_4(self.up_sampling1d_4(x)[:, :, 1:-1])      # cropping
        x = self.conv1d_5(self.up_sampling1d_5(x))
        x = self.conv1d_6(self.up_sampling1d_6(x))
        x = self.conv1d_7(x)
        x = self.conv1d_8(x)
        x = self.fc(x).permute(0, 2, 1)
        return x


class EQTransformer(nn.Module):
    def __init__(self):
        super(EQTransformer, self).__init__()
        self.encoder = EQTransformerEncoder()
        self.decoder = EQTransformerDecoder()

    def forward(self, inputs):
        encoder_outputs = self.encoder(inputs)
        outputs = self.decoder(encoder_outputs)
        return outputs


class Yews(nn.Module):
    def __init__(self):
        super(Yews, self).__init__()
        self.conv1d_1 = nn.Conv1d(3, 16, 5, 1, padding=2)
        self.max_pooling1d_1 = nn.MaxPool1d(2)
        self.conv1d_2 = nn.Conv1d(16, 32, 5, 1, padding=2)
        self.max_pooling1d_2 = nn.MaxPool1d(2)
        self.conv1d_3 = nn.Conv1d(32, 64, 3, 1, padding=1)
        self.max_pooling1d_3 = nn.MaxPool1d(2)
        self.conv1d_4 = nn.Conv1d(64, 64, 3, 1, padding=4)
        self.max_pooling1d_4 = nn.MaxPool1d(2)
        self.conv1d_5 = nn.Conv1d(64, 64, 3, 1, padding=1)
        self.max_pooling1d_5 = nn.MaxPool1d(2)
        self.conv1d_6 = nn.Conv1d(64, 64, 3, 1, padding=1)
        self.max_pooling1d_6 = nn.MaxPool1d(2)
        self.conv1d_7 = nn.Conv1d(64, 64, 3, 1, padding=1)
        self.max_pooling1d_7 = nn.MaxPool1d(2)
        self.conv1d_8 = nn.Conv1d(64, 64, 3, 1, padding=1)
        self.max_pooling1d_8 = nn.MaxPool1d(2)
        self.conv1d_9 = nn.Conv1d(64, 64, 3, 1, padding=1)
        self.max_pooling1d_9 = nn.MaxPool1d(2)
        self.conv1d_10 = nn.Conv1d(64, 64, 3, 1, padding=1)
        self.max_pooling1d_10 = nn.MaxPool1d(2)
        self.conv1d_11 = nn.Conv1d(64, 64, 3, 1, padding=1)
        self.max_pooling1d_11 = nn.MaxPool1d(2)
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        x = x[:, :, 2000:4000]  # cropping
        x = self.max_pooling1d_1(self.conv1d_1(x))
        x = self.max_pooling1d_2(self.conv1d_2(x))
        x = self.max_pooling1d_3(self.conv1d_3(x))
        x = self.max_pooling1d_4(self.conv1d_4(x))
        x = self.max_pooling1d_5(self.conv1d_5(x))
        x = self.max_pooling1d_6(self.conv1d_6(x))
        x = self.max_pooling1d_7(self.conv1d_7(x))
        x = self.max_pooling1d_8(self.conv1d_8(x))
        x = self.max_pooling1d_9(self.conv1d_9(x))
        x = self.max_pooling1d_10(self.conv1d_10(x))
        x = self.max_pooling1d_11(self.conv1d_11(x)).permute(0, 2, 1)
        x = self.fc(x).permute(0, 2, 1)
        return x


class CREDFeatureExtractor(nn.Module):
    def __init__(self):
        super(CREDFeatureExtractor, self).__init__()

    def forward(self, x):
        E, N, Z = x[:, 0].cpu(), x[:, 1].cpu(), x[:, 2].cpu()
        E = torch.Tensor(make_spectrogram(E)).unsqueeze(1)
        N = torch.Tensor(make_spectrogram(N)).unsqueeze(1)
        Z = torch.Tensor(make_spectrogram(Z)).unsqueeze(1)
        output = torch.cat([E, N, Z], dim=1).cuda()
        return output


class CREDEncoder(nn.Module):
    def __init__(self):
        super(CREDEncoder, self).__init__()
        self.conv2d_1 = nn.Conv2d(3, 8, 9, 2)
        self.batch_normalization_1 = nn.BatchNorm2d(8)
        self.activation_1 = nn.ReLU()
        self.conv2d_2 = nn.Conv2d(8, 8, 7, 1, padding='same')
        self.batch_normalization_2 = nn.BatchNorm2d(8)
        self.activation_2 = nn.ReLU()
        self.conv2d_3 = nn.Conv2d(8, 8, 7, 1, padding='same')
        self.conv2d_4 = nn.Conv2d(8, 16, 5, 2)
        self.batch_normalization_3 = nn.BatchNorm2d(16)
        self.activation_3 = nn.ReLU()
        self.conv2d_5 = nn.Conv2d(16, 16, 3, 1, padding='same')
        self.batch_normalization_4 = nn.BatchNorm2d(16)
        self.activation_4 = nn.ReLU()
        self.conv2d_6 = nn.Conv2d(16, 16, 3, 1, padding='same')

    def forward(self, x):
        x = self.conv2d_1(x)
        res = x
        x = self.activation_1(self.batch_normalization_1(x))
        x = self.conv2d_2(x)
        x = self.activation_2(self.batch_normalization_2(x)) + res
        x = self.conv2d_3(x)
        x = self.conv2d_4(x)
        res = x
        x = self.activation_3(self.batch_normalization_3(x))
        x = self.conv2d_5(x)
        x = self.activation_4(self.batch_normalization_4(x))
        x = self.conv2d_6(x) + res
        return x


class CREDDecoder(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 bias: bool = True,
                 batch_first: bool = True,
                 bidirectional: bool = True):
        super(CREDDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.lstm1 = nn.LSTM(input_size=input_size,
                             hidden_size=hidden_size,
                             num_layers=num_layers,
                             bias=bias,
                             batch_first=batch_first,
                             bidirectional=bidirectional)

        self.lstm2 = nn.LSTM(input_size=input_size * 2,
                             hidden_size=hidden_size * 2,
                             num_layers=1,
                             bias=bias,
                             batch_first=batch_first,
                             bidirectional=False)

        self.fc1 = nn.Linear(in_features=input_size * 2,
                             out_features=input_size * 2)
        self.fc2 = nn.Linear(in_features=input_size * 2,
                             out_features=1)
        self.fc3 = nn.Linear(in_features=16,
                             out_features=2)

    def forward(self, x):
        x = x.view(-1, 16, 130)
        x, (_, _) = self.lstm1(x)
        x, (_, _) = self.lstm2(x)
        x = self.fc1(x)
        x = self.fc2(x).permute(0, 2, 1)
        x = self.fc3(x).permute(0, 2, 1)
        return x


class CRED(nn.Module):
    def __init__(self):
        super(CRED, self).__init__()
        self.feature_extractor = CREDFeatureExtractor()
        self.encoder = CREDEncoder()
        self.decoder = CREDDecoder(input_size=130,
                                   hidden_size=130,
                                   num_layers=2)

    def forward(self, inputs):
        inputs = self.feature_extractor(inputs)
        encoder_outputs = self.encoder(inputs)
        outputs = self.decoder(encoder_outputs)
        return outputs


class DetNet(nn.Module):
    def __init__(self):
        super(DetNet, self).__init__()
        self.conv1d_1 = nn.Conv1d(3, 32, 3, 1, padding='same')
        self.activation_1 = nn.ReLU()
        self.max_pooling1d_1 = nn.MaxPool1d(2)
        self.CRPlayer = nn.Sequential(nn.Conv1d(32, 32, 3, 1, padding='same'),
                                      nn.ReLU(),
                                      nn.MaxPool1d(2))
        self.fc = nn.Linear(32 * 23, 2)

    def forward(self, x):
        x = self.max_pooling1d_1(self.activation_1(self.conv1d_1(x)))
        for i in range(7):
            x = self.CRPlayer(x)
        x = self.fc(x.view(-1, 32 * 23)).unsqueeze(2)
        return x


class DeepConvBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int):
        super(DeepConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sequential_1 = nn.Sequential(nn.Conv1d(in_channels=in_channels,
                                                    out_channels=out_channels,
                                                    kernel_size=kernel_size,
                                                    stride=1),
                                          nn.LeakyReLU())

        self.sequential_2 = nn.Sequential(nn.Conv1d(in_channels=out_channels,
                                                    out_channels=out_channels,
                                                    kernel_size=kernel_size,
                                                    stride=1),
                                          nn.LeakyReLU())
        self.sequential_3 = nn.Sequential(nn.Conv1d(in_channels=out_channels,
                                                    out_channels=out_channels,
                                                    kernel_size=kernel_size,
                                                    stride=1),
                                          nn.LeakyReLU())

        self.sequential_4 = nn.Sequential(nn.Conv1d(in_channels=out_channels,
                                                    out_channels=out_channels,
                                                    kernel_size=kernel_size,
                                                    stride=1),
                                          nn.MaxPool1d(2),
                                          nn.LeakyReLU())

        self.dropout_1 = nn.Dropout1d(p=0.2)

    def forward(self, x):
        x = self.sequential_1(x)
        x = self.sequential_2(x)
        x = self.dropout_1(x)
        x = self.sequential_3(x)
        x = self.sequential_4(x)
        x = self.dropout_1(x)
        return x


class DeeperCRNN(nn.Module):
    def __init__(self):
        super(DeeperCRNN, self).__init__()

        self.embedding_block = nn.Sequential(nn.Linear(1, 16),
                                             nn.Linear(16, 16),
                                             nn.Linear(16, 16),
                                             nn.Linear(16, 1))

        self.deep_conv_block_1 = DeepConvBlock(in_channels=3,
                                               out_channels=128,
                                               kernel_size=9)
        self.deep_conv_block_2 = DeepConvBlock(in_channels=128,
                                               out_channels=128,
                                               kernel_size=7)
        self.deep_conv_block_3 = DeepConvBlock(in_channels=128,
                                               out_channels=128,
                                               kernel_size=5)
        self.deep_conv_block_4 = DeepConvBlock(in_channels=128,
                                               out_channels=128,
                                               kernel_size=3)

        self.lstm = nn.LSTM(input_size=362,
                            hidden_size=100,
                            bidirectional=True)
        self.estimator = nn.Sequential(nn.Linear(201, 128),
                                       nn.ReLU(),
                                       nn.Dropout1d(p=0.5),
                                       nn.Linear(128, 128),
                                       nn.Linear(128, 1))

    def forward(self, x):
        max_amplitude = torch.Tensor(
            [max([max(x[i, j, :]) for j in range(2)]) for i in range(len(x[:, 0, 0]))]).unsqueeze(1).cuda()
        embedded_x = self.embedding_block(max_amplitude).cuda()
        x = self.deep_conv_block_1(x)
        x = self.deep_conv_block_2(x)
        x = self.deep_conv_block_3(x)
        x = self.deep_conv_block_4(x)
        x, (_, _) = self.lstm(x)
        x = torch.cat([x[:, -1, :], embedded_x], dim=1)
        x = self.estimator(x)
        return x


class MagNet(nn.Module):
    def __init__(self):
        super(MagNet, self).__init__()
        self.encoder = nn.Sequential(nn.Conv1d(3, 64, 3, 1),
                                     nn.Dropout1d(p=0.2),
                                     nn.MaxPool1d(4),
                                     nn.Conv1d(64, 32, 3, 1),
                                     nn.Dropout1d(p=0.2),
                                     nn.MaxPool1d(4))
        self.decoder = nn.LSTM(input_size=374, hidden_size=100, bidirectional=True)
        self.fc = nn.Linear(200, 1)

    def forward(self, x):
        x = self.encoder(x)
        x, (_, _) = self.decoder(x)
        x = self.fc(x[:, -1, :])
        return x
