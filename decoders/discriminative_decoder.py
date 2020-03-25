#!/usr/bin/env python
__author__ = "Mohamed Adnen Abdessaied"
__version__ = "1.0"
__maintainer__ = "Mohamed Adnen Abdessaied"
__email__ = "adnenabdessayed@gmail.com"
__status__ = "Implementation"

import torch
import torch.nn as nn

from preprocessing.statics import (
    EMBEDDING_DIM,
    VOCAB_SIZE,
    ANSWER_OPTS
)
from decoders.statics import (
    LSTM_HIDDEN_SIZE,
    LSTM_NUM_LAYERS,
    DROPOUT
)


class Disc_Decoder(nn.Module):
    """
    This class implements the discriminative decoder as described in visual dialog paper:
        arXiv: 1611.08669v5
    """

    def __init__(self, params: dict):
        """
        Class constructor.
        :param params:  The parameters of the network.
        :param embedding: the embedding layer shared between the encoder and decoder nets.
        """
        super(Disc_Decoder, self).__init__()
        self.params = params
        self.embedding = nn.Embedding(params[VOCAB_SIZE], params[EMBEDDING_DIM], padding_idx=0)

        self.answer_opts_lstm = nn.LSTM(self.params[EMBEDDING_DIM],
                                        self.params[LSTM_HIDDEN_SIZE],
                                        self.params[LSTM_NUM_LAYERS],
                                        batch_first=True,
                                        dropout=self.params[DROPOUT])

    def forward(self, encoder_output: torch.Tensor, batch: dict):
        """
        Implementation of the forward pass of the decoder.
        :param encoder_output: The output of the decoder. Shape : (batch_size, rounds, lstm_hidden_size)
        :param batch: The batch of data as defined in our "Visual_Dialog_Dataset"
        :return: The encoder_output as a torch.Tensor
        """

        # Shape: (batch_size, rounds, num_opts, max_sentence_length)
        answer_opts = batch[ANSWER_OPTS]
        batch_size, rounds, num_opts, max_sentence_length = answer_opts.size()

        # Shape: (batch_size * rounds * num_opts, max_sentence_length)
        answer_opts = answer_opts.view(-1, max_sentence_length)

        # Shape: (batch_size * rounds * num_opts, max_sentence_length, Embedding_dim)
        answer_opts_embedded = self.embedding(answer_opts)

        # Shape: (lstm_num_layers, batch_size * rounds * num_opts, lstm_hidden_size)
        _, (answer_opts_embedded, _) = self.answer_opts_lstm(answer_opts_embedded)

        # Shape: (batch_size * rounds * num_opts, lstm_hidden_size)
        answer_opts_embedded = answer_opts_embedded[-1]

        # We compute the similarity between the input encoding and the LSTM encoding of each answer.
        # Therefore we need to repeat the encoder output 100 (# of answer options).

        # Shape: (batch_size * rounds * num_opts, lstm_hidden_size)
        encoder_output = encoder_output.unsqueeze(2).repeat(1, 1, num_opts, 1).view(-1, self.params[LSTM_HIDDEN_SIZE])

        # Shape: (batch_size * rounds * num_opts)
        similarities = torch.sum(encoder_output * answer_opts_embedded, dim=1)

        # Shape: (batch_size, rounds, num_opts)
        return similarities.view(batch_size, rounds, num_opts)
