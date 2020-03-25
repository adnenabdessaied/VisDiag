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


class Memory_Network(nn.Module):
    """
    This class implements the memory network encoder as described in the visual dialog paper:
        arXiv: 1611.08669v5
    """
    def __init__(self, params: dict):
        """
        Class constructor.
        :param params: The parameters of the network.
        :param path_to_embedding_matrix: Path to the pre-trained embedding matrix/weights.
        """
        super(Memory_Network, self).__init__()
        self.params = params
        self.embedding = nn.Embedding(params[VOCAB_SIZE], params[EMBEDDING_DIM], padding_idx=PAD_INDEX)

        self.question_lstm = nn.LSTM(self.params[EMBEDDING_DIM],
                                     self.params[LSTM_HIDDEN_SIZE],
                                     self.params[LSTM_NUM_LAYERS],
                                     batch_first=True,
                                     dropout=self.params[DROPOUT])

        self.fc_1 = nn.Linear(self.params[IMG_FEATURE_DIM] + self.params[LSTM_HIDDEN_SIZE],
                              self.params[LSTM_HIDDEN_SIZE])

        # Each fact, i.e. a caption or a qa-pair, will be encoded independently using an LSTM with shared weights. Thus,
        # the same LSTM will be used separately to encode each fact.
        self.fact_lstm = nn.LSTM(self.params[EMBEDDING_DIM],
                                 self.params[LSTM_HIDDEN_SIZE],
                                 self.params[LSTM_NUM_LAYERS],
                                 batch_first=True,
                                 dropout=self.params[DROPOUT])

        self.softmax = nn.Softmax(dim=-1)
        self.fc_2 = nn.Linear(self.params[LSTM_HIDDEN_SIZE], self.params[LSTM_HIDDEN_SIZE])
        self.dropout = nn.Dropout(p=self.params[DROPOUT])

        # Initialization of the weights and biases of the first fully connected layer
        nn.init.kaiming_uniform_(self.fc_1.weight)
        nn.init.constant_(self.fc_1.bias, 0)

        # Initialization of the weights and biases of the second fully connected layer
        nn.init.kaiming_uniform_(self.fc_2.weight)
        nn.init.constant_(self.fc_2.bias, 0)

    def forward(self, batch: dict):
        """
        Implements the forward pass of the memory network encoder.
        :param batch: Batch of data as defined in our "Visual_Dialog_Dataset"
        :return:
        """
        # Shape: (batch_size, 1, image_feature_dim)
        img_features = batch[IMG_FEATURES]

        # Shape: (batch_size, rounds, MAX_SENTENCE_LENGTH)
        questions = batch[QUESTIONS]

        # Shape: (batch_size, rounds, MAX_SENTENCE_LENGTH * 2)
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

        # Repeat the image features for each dialog round
        img_features = img_features.repeat(1, rounds, 1).view(
            batch_size * rounds, -1, self.params[IMG_FEATURE_DIM]).squeeze()

        # Concatenate the image features and the lstm encoding of the questions
        concatenated_rep = torch.cat([img_features.float(), questions_embeddings_hidden], dim=1)

        # Get the "query_vectors"
        query_vectors = torch.tanh(self.fc_1(concatenated_rep))
        query_vectors = self.dropout(query_vectors)

        attended_histories = []
        # Embedding of the history facts
        for i in range(batch_size):
            batch_history = history[i, :, :]
            batch_query = query_vectors[i * rounds: (i+1) * rounds, :]
            history_encoded = []
            for j in range(rounds):
                # Shape: (1, MAX_SENTENCE_LENGTH * 2)
                fact_t = batch_history[j: j+1, :]
                # Shape: (1, MAX_SENTENCE_LENGTH * 2, Embedding_dim)
                fact_t_embedding = self.embedding(fact_t)
                # Shape: (lstm_num_layers, 1, lstm_hidden_size)
                _, (fact_t_embedding_hidden, _) = self.fact_lstm(fact_t_embedding)
                # Shape: (1, lstm_hidden_size)
                fact_t_embedding_hidden = fact_t_embedding_hidden[-1]
                history_encoded.append(fact_t_embedding_hidden)

            # Stack the history encodings in a torch tensor
            history_encoded = torch.cat(history_encoded, dim=0)

            # compute the attention weights
            for j in range(rounds):
                query_vectors_repeated = batch_query[j: j+1, :].repeat(j+1, 1)
                relevant_history = history_encoded[: j+1, :]
                inner_prod = query_vectors_repeated * relevant_history
                attention_weights = inner_prod.sum(dim=1)

                # Use a softmax layer to normalize the attention scores
                attention_weights = self.softmax(attention_weights)
                attended_history = torch.zeros((1, self.params[LSTM_HIDDEN_SIZE]), device=relevant_history.device)
                for k in range(j + 1):
                    attended_history += relevant_history[k, :] * attention_weights[k].item()
                attended_histories.append(attended_history)
        attended_histories = torch.cat(attended_histories, dim=0)
        attended_histories = torch.tanh(self.fc_2(attended_histories))
        attended_histories = self.dropout(attended_histories)
        encoder_output = query_vectors + attended_histories

        # Shape: (batch_size, rounds, lstm_hidden_size)
        return encoder_output.view(batch_size, rounds, -1)
