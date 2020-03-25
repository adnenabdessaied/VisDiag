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
    ANSWERS_GEN_TR,
    ANSWER_OPTS_GEN_TR,
    ANSWER_OPTS_GEN_TARGETS,
    SOS_INDEX,
    EOS_INDEX,
    UNK_TOKEN
)
from decoders.statics import (
    LSTM_HIDDEN_SIZE,
    LSTM_NUM_LAYERS,
    DROPOUT,

)


class Gen_Decoder(nn.Module):
    """
    This class implements the generative decoder as described in visual dialog paper:
    arXiv: 1611.08669v5
    """
    def __init__(self, params: dict, idx_to_word):
        """
        Class constructor.
        :param params:  The parameters of the network.
        :param embedding: the embedding layer shared between the encoder and decoder nets.
        :param idx_to_word: The index to word mapping of our vocabulary.
        """
        super(Gen_Decoder, self).__init__()
        self.params = params
        self.embedding = nn.Embedding(params[VOCAB_SIZE], params[EMBEDDING_DIM], padding_idx=0)
        self.idx_to_word = idx_to_word
        self.gen_answer_lstm = nn.LSTM(self.params[EMBEDDING_DIM],
                                       self.params[LSTM_HIDDEN_SIZE],
                                       self.params[LSTM_NUM_LAYERS],
                                       batch_first=True,
                                       dropout=self.params[DROPOUT])

        self.words_from_lstm = nn.Linear(self.params[LSTM_HIDDEN_SIZE], self.params[VOCAB_SIZE])
        self.dropout = nn.Dropout(p=self.params[DROPOUT])
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        # Initialization of the weights and biases
        nn.init.kaiming_uniform_(self.words_from_lstm.weight)
        nn.init.constant_(self.words_from_lstm.bias, 0)

    def forward(self, encoder_output: torch.Tensor, batch: dict, inference=False):
        """
        Implementation of the forward pass of the decoder.
        :param encoder_output: The input of the decoder. Shape : (batch_size, 10, lstm_hidden_size)
        :param batch: The batch of data as defined in our "Visual_Dialog_Dataset"
        :param inference: If true, the generated dialog will be returned instead of the scores. We use this feature
        in the interactive mode (inference).
        :return: The encoder_output or the generated dialog.
        """
        if self.training:
            # Shape: (batch_size, rounds, max_sentence_length)
            answer_gen_tr = batch[ANSWERS_GEN_TR]

            batch_size, rounds, max_sentence_length = answer_gen_tr.size()

            # Reshape it to (batch_size * rounds, max_sentence_length)
            answer_gen_tr = answer_gen_tr.view(-1, max_sentence_length)

            # Generate the embedding of the answers
            # Shape: (batch_size * rounds, max_sentence_length, lstm_hidden_size)
            answer_gen_tr_embed = self.embedding(answer_gen_tr)

            # The hidden state of an lstm must have a shape of (lstm_num_layers, batch_size, lstm_hidden_size).
            # We want to use the encoder_output as the initial hidden state of the lstm cell. Thus, reshaping it
            # to the desired shape is mandatory.
            # The initial shape of the output tensor is (batch_size, rounds, lstm_hidden_size)
            h_0 = encoder_output.view(batch_size * rounds, -1)

            # Shape: (1, batch_size * rounds, lstm_hidden_size)
            h_0 = h_0.unsqueeze(0)

            # Shape: (lstm_num_layers, batch_size * rounds, lstm_hidden_size)
            h_0 = h_0.repeat(self.params[LSTM_NUM_LAYERS], 1, 1)

            # Initialization of the cell state variable
            c_0 = torch.zeros_like(h_0)

            # Shape: (batch_size * rounds, max_sentence_length, lstm_hidden_size)
            lstm_output, (_, _) = self.gen_answer_lstm(answer_gen_tr_embed, (h_0, c_0))
            lstm_output = self.dropout(lstm_output)

            # Get words from lstm
            # Shape: (batch_size * rounds, max_sentence_length, vocab_size)
            scores = self.words_from_lstm(lstm_output)
            return scores
        else:
            try:
                ans_in = batch[ANSWER_OPTS_GEN_TR]
                batch_size, num_rounds, num_options, max_sequence_length = (ans_in.size())

                ans_in = ans_in.view(batch_size * num_rounds * num_options, max_sequence_length)

                # shape: (batch_size * num_rounds * num_options, max_sequence_length, word_embedding_size)

                ans_in_embed = self.embedding(ans_in)

                # reshape encoder output to be set as initial hidden state of LSTM.

                # shape: (lstm_num_layers, batch_size * num_rounds * num_options,

                #         lstm_hidden_size)

                init_hidden = encoder_output.view(batch_size, num_rounds, 1, -1)

                init_hidden = init_hidden.repeat(1, 1, num_options, 1)

                init_hidden = init_hidden.view(1, batch_size * num_rounds * num_options, -1)

                init_hidden = init_hidden.repeat(self.params[LSTM_NUM_LAYERS], 1, 1)
                init_cell = torch.zeros_like(init_hidden)

                # shape: (batch_size * num_rounds * num_options,

                #         max_sequence_length, lstm_hidden_size)

                ans_out, (hidden, cell) = self.gen_answer_lstm(ans_in_embed, (init_hidden, init_cell))

                # shape: (batch_size * num_rounds * num_options,

                #         max_sequence_length, vocabulary_size)

                ans_word_scores = self.logsoftmax(self.words_from_lstm(ans_out))

                # shape: (batch_size * num_rounds * num_options,

                #         max_sequence_length)

                target_ans_out = batch[ANSWER_OPTS_GEN_TARGETS].view(batch_size * num_rounds * num_options, -1)

                # shape: (batch_size * num_rounds * num_options,

                #         max_sequence_length)

                ans_word_scores = torch.gather(ans_word_scores, -1, target_ans_out.unsqueeze(-1)).squeeze()

                ans_word_scores = (ans_word_scores * (target_ans_out > 0).float().to(encoder_output.device))  # ugly

                ans_scores = torch.sum(ans_word_scores, -1)

                ans_scores = ans_scores.view(batch_size, num_rounds, num_options)
            except KeyError:
                ans_scores = []

            generated_dialog = self.predict_dialog(encoder_output)
            return ans_scores, generated_dialog

    def predict_dialog(self, encoder_output, start_idx=SOS_INDEX, answer_length=10):
        self.eval()
        batch_size, rounds, lstm_hidden_size = encoder_output.size()
        generated_answers = []
        for i in range(rounds):
            answer_per_round = []
            # Shape: (batch_size, 1, lstm_hidden_size)
            encoding_round_i = encoder_output[:, i:i + 1, :]

            # Sahpe: (batch_size, lstm_hidden_size)
            h_0 = encoding_round_i.contiguous().view(batch_size, -1)

            # Sahpe: (1, batch_size, lstm_hidden_size)
            h_0 = h_0.unsqueeze(0)

            # Shape: (lstm_num_layers, batch_size * (i + 1), lstm_hidden_size)
            h_0 = h_0.repeat(self.params[LSTM_NUM_LAYERS], 1, 1)

            # Initialization of the cell state variable
            c_0 = torch.zeros_like(h_0)

            # Get the embedding of the first word that we will use to generate the answer.
            # Shape: (1, embedding_size)
            x_0 = self.embedding(torch.tensor([start_idx]).long().to(h_0.device))

            # Shape: (1, 1, embedding_size)
            x_0 = x_0.unsqueeze(0)

            # Shape: (batch_size, 1, embedding_size)
            x_0 = x_0.repeat(batch_size, 1, 1)

            # get the output of the lstm:
            output, (h_0, c_0) = self.gen_answer_lstm(x_0, (h_0, c_0))

            # Shape: (batch_size, vocab_size)
            scores = self.words_from_lstm(output).squeeze(1)
            _, top_1_indices = torch.topk(scores, 1, dim=-1)

            # choices = torch.tensor([top_k_indices[i, rand_idx[i]] for i in range(batch_size)]).to(h_0.device)
            answer_per_round.append(top_1_indices)
            for j in range(answer_length - 1):
                # For the next words, we use the previously predicted one as input to the lstm
                # Shape: (batch_size, 1, embedding_size)
                x_0 = self.embedding(top_1_indices)

                # get the output of the lstm:
                output, (h_0, c_0) = self.gen_answer_lstm(x_0, (h_0, c_0))

                # Shape: (batch_size, vocab_size)
                scores = self.words_from_lstm(output).squeeze(1)
                _, top_1_indices = torch.topk(scores, 1, dim=-1)
                answer_per_round.append(top_1_indices)
            generated_answers.append(torch.cat(answer_per_round, dim=1))

        dialogs = self.from_idx_to_words(generated_answers, self.idx_to_word)
        return dialogs

    @staticmethod
    def from_idx_to_words(generated_answers, idx_to_word):
        """
        Returns the answers generated by the network.
        :param generated_answers: Indices of the generated words.
        :param idx_to_word: The index to word mapping.
        :return: List of full dialogs. The first element holds all 10 answers of 1 image.
        """
        batch_size, _ = generated_answers[0].squeeze(-1).size()
        answers = []
        for diag_round in generated_answers:
            answers_round_i = []
            for i in range(batch_size):
                current_answer_batch = diag_round[i, :]
                current_answer_batch = current_answer_batch.tolist()
                answer = []

                # Index to word mapping
                for idx in current_answer_batch:
                    if idx == EOS_INDEX:
                        break
                    else:
                        answer.append(idx_to_word.get(idx, UNK_TOKEN))
                answers_round_i.append(" ".join(answer))

            answers.append(answers_round_i)

        dialogs = []
        for i in range(batch_size):
            dialog = []
            for diag_round in answers:
                dialog.append(diag_round[i])
            dialogs.append(dialog)
        # dialogs = [[answer_1, ..., answer_10],..., [answer_1, ..., answer_10]]
        return dialogs
