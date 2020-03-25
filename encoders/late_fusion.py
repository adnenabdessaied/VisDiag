#!/usr/bin/env python
__author__ = "Mohamed Adnen Abdessaied"
__version__ = "1.0"
__maintainer__ = "Mohamed Adnen Abdessaied"
__email__ = "adnenabdessayed@gmail.com"
__status__ = "Implementation"

import torch
import torch.nn as nn
import torch.nn.functional as F

from preprocessing.statics import (
    EMBEDDING_DIM,
    VOCAB_SIZE,
    IMG_FEATURES,
    QUESTIONS,
    HISTORY,
    PAD_INDEX
)
from encoders.statics import (
    LSTM_HIDDEN_SIZE,
    LSTM_NUM_LAYERS,
    DROPOUT,
    IMG_FEATURE_DIM
)


class Late_Fusion(nn.Module):
    """
    This class implements the late fusion encoder as described in visual the dialog paper:
        arXiv: 1611.08669v5
    """

    def __init__(self, params: dict):
        """
        Class constructor.
        :param params: The parameters of the network.
        :param path_to_embedding_matrix: Path to the pre-trained embedding matrix/weights.
        """
        super(Late_Fusion, self).__init__()
        self.params = params
        self.embedding = nn.Embedding(params[VOCAB_SIZE], params[EMBEDDING_DIM], padding_idx=PAD_INDEX)
        self.question_lstm = nn.LSTM(self.params[EMBEDDING_DIM],
                                     self.params[LSTM_HIDDEN_SIZE],
                                     self.params[LSTM_NUM_LAYERS],
                                     batch_first=True,
                                     dropout=self.params[DROPOUT])
        self.history_lstm = nn.LSTM(self.params[EMBEDDING_DIM],
                                    self.params[LSTM_HIDDEN_SIZE],
                                    self.params[LSTM_NUM_LAYERS],
                                    batch_first=True,
                                    dropout=self.params[DROPOUT])

        # In order to implement the attention mechanism, we need to project the image features to have the  same
        # dimension as the lstm hidden size. The questions are used to carry out attention on image_features.
        self.img_projection = nn.Linear(self.params[IMG_FEATURE_DIM], self.params[LSTM_HIDDEN_SIZE]).double()

        self.attention = nn.Linear(self.params[LSTM_HIDDEN_SIZE], 1)

        # At the end, we concatenate the image features, the LSTM encoding of questions and dialog histories to form
        # the final fusion.
        fusion_dim = self.params[IMG_FEATURE_DIM] + 2 * self.params[LSTM_HIDDEN_SIZE]

        # Dropout
        self.dropout = nn.Dropout(p=self.params[DROPOUT])

        # The last step of the encoder
        self.fusion = nn.Linear(fusion_dim, self.params[LSTM_HIDDEN_SIZE])

        # Initialization of the weights and biases
        nn.init.kaiming_uniform_(self.img_projection.weight)
        nn.init.constant_(self.img_projection.bias, 0)

        nn.init.kaiming_uniform_(self.fusion.weight)
        nn.init.constant_(self.fusion.bias, 0)

    def forward(self, batch: dict):
        """
        Implements the forward pass of the late fusion encoder.
        :param batch: Batch of data as defined in our "Visual_Dialog_Dataset"
        :return:
        """
        # Shape: (batch_size, 1, image_feature_dim)
        img_features = batch[IMG_FEATURES]

        # Shape: (batch_size, rounds, max_sentence_length)
        questions = batch[QUESTIONS]

        # Shape: (batch_size, rounds, max_sentence_length * 2)
        history = batch[HISTORY]
        batch_size, rounds, max_sentence_length = questions.size()
        # Reshape questions to fit into the embedding --> Shape: (batch_size * rounds, max_sentence_length)
        questions = questions.view(-1, max_sentence_length)

        # Shape: (batch_size * rounds, max_sentence_length, Embedding_dim)
        questions_embeddings = self.embedding(questions)

        # Shape: (lstm_num_layers, batch_size * rounds, lstm_hidden_size)
        _, (questions_embeddings_hidden, _) = self.question_lstm(questions_embeddings)

        # Shape: (batch_size * rounds, lstm_hidden_size)
        questions_embeddings_hidden = questions_embeddings_hidden[-1]
        # Shape: (batch_size, lstm_hidden_size)
        img_features_projected = self.img_projection(img_features)

        # Image features vector should be available in each round in order to be able to implement the attention
        # mechanism.
        # Shape: (batch_size * rounds, lstm_hidden_size)
        img_features_projected = img_features_projected.repeat(1, rounds, 1).view(
            batch_size * rounds, -1, self.params[LSTM_HIDDEN_SIZE]).squeeze()

        # Compute the image attention weights
        image_question_projected = img_features_projected.float() * questions_embeddings_hidden
        img_attention_weights = F.softmax(self.attention(image_question_projected), dim=0)

        # Shape: (batch_size * rounds, self.params[IMG_FEATURE_DIM])
        img_attention_weights = img_attention_weights.repeat(1, self.params[IMG_FEATURE_DIM])

        img_features = img_features.repeat(1, rounds, 1).view(
            batch_size * rounds, -1, self.params[IMG_FEATURE_DIM]).squeeze()

        # Multiply each image with its weight
        # Shape: (batch_size * rounds, self.params[IMG_FEATURE_DIM])
        img_features_attended = img_features.float() * img_attention_weights

        # Embedding of the history
        history = history.view(batch_size * rounds, -1)

        # Shape: (batch_size * rounds, max_sentence_length * 2, Embedding_dim)
        history_embedding = self.embedding(history)

        # Shape: (lstm_num_layers, batch_size * rounds, lstm_hidden_size)
        _, (history_embedding_hidden, _) = self.question_lstm(history_embedding)

        # Shape: (batch_size * rounds, lstm_hidden_size)
        history_embedding_hidden = history_embedding_hidden[-1]

        # Concatenate all these encodings to form the fusion vector
        # Shape: (batch_size * rounds, fusion_dim)
        fusion = torch.cat([img_features_attended,
                            questions_embeddings_hidden,
                            history_embedding_hidden],
                           dim=1)
        fusion = self.dropout(fusion)
        fusion_embedded = torch.tanh(self.fusion(fusion))

        # Shape: (batch_size, rounds, lstm_hidden_size)
        return fusion_embedded.view(batch_size, rounds, -1)
