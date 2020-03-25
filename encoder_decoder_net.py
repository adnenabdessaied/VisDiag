#!/usr/bin/env python
__author__ = "Mohamed Adnen Abdessaied"
__version__ = "1.0"
__maintainer__ = "Mohamed Adnen Abdessaied"
__email__ = "adnenabdessayed@gmail.com"
__status__ = "Implementation"

import torch.nn as nn


class EncoderDecoderNet(nn.Module):
    """
    A helper class that condenses the encoder-decoder architecture
    """
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        """
        Class Constructor.
        :param encoder: The encoder network.
        :param decoder: The decoder network
        """
        super(EncoderDecoderNet, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, batch):
        """
        Implementation of the forward pass of the network.
        :param batch: Batch of data as defined in our "Visual_Dialog_Dataset"
        :return: The decoder output
        """
        encoder_output = self.encoder(batch)
        decoder_output = self.decoder(encoder_output, batch)
        return decoder_output

