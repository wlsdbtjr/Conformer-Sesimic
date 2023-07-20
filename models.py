import torch.nn as nn
import loralib as lora
from utils import LoRAize
from conformer.encoder import ConformerEncoder
from modules import SpectrogramENZ, ClassificationDecoder, MagnitudeEstimationDecoder


class SeismicEventClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(SeismicEventClassifier, self).__init__()
        self.num_classes = num_classes
        self.feature_extractor = SpectrogramENZ()
        self.encoder = ConformerEncoder(
                                        input_dim=65,
                                        encoder_dim=160,
                                        num_layers=6,
                                        num_attention_heads=8,
                                        feed_forward_expansion_factor=4,
                                        conv_expansion_factor=2,
                                        input_dropout_p=0.1,
                                        feed_forward_dropout_p=0.1,
                                        attention_dropout_p=0.1,
                                        conv_dropout_p=0.1,
                                        conv_kernel_size=31,
                                        half_step_residual=True)
        self.decoder = ClassificationDecoder(dim=160)

    def forward(self, inputs):
        inputs = self.feature_extractor(inputs)
        encoder_outputs = self.encoder(inputs)
        outputs = self.decoder(encoder_outputs)
        return outputs


class MagnitudeEstimator(nn.Module):
    def __init__(self):
        super(MagnitudeEstimator, self).__init__()
        self.feature_extractor = SpectrogramENZ()
        self.encoder = ConformerEncoder(
                                        input_dim=65,
                                        encoder_dim=160,
                                        num_layers=6,
                                        num_attention_heads=8,
                                        feed_forward_expansion_factor=4,
                                        conv_expansion_factor=2,
                                        input_dropout_p=0.1,
                                        feed_forward_dropout_p=0.1,
                                        attention_dropout_p=0.1,
                                        conv_dropout_p=0.1,
                                        conv_kernel_size=31,
                                        half_step_residual=True)
        self.decoder = MagnitudeEstimationDecoder(input_size=160,
                                                  hidden_size=160,
                                                  num_layers=4)

    def forward(self, inputs):
        inputs = self.feature_extractor(inputs)
        encoder_outputs = self.encoder(inputs)
        outputs = self.decoder(encoder_outputs)
        return outputs