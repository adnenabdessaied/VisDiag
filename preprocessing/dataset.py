#!/usr/bin/env python
__author__ = "Mohamed Adnen Abdessaied"
__version__ = "1.0"
__maintainer__ = "Mohamed Adnen Abdessaied"
__email__ = "adnenabdessayed@gmail.com"
__status__ = "Implementation"

"""
This class implements a dataset that will be used afterwards for training. We use here the functionalities of 
data readers and vocabulary readers
"""

import os
import pickle
import torch

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset
from preprocessing.data_readers import Reader
from preprocessing.vocabulary_reader import Vocabulary_Reader
from feature_extractors.custom_nets import VGG_clipped, Googlenet_clipped
from num2words import num2words

from preprocessing.statics import (
    IMAGE_PATHS,
    IMAGE_ID,
    CAPTIONS,
    DIALOGS,
    ANSWERS,
    ANSWER,
    QUESTIONS,
    QUESTION,
    MAX_SENTENCE_LENGTH,
    HISTORY,
    IMG_FEATURES,
    ANSWER_OPTS,
    ANSWER_GT,
    DENSE_ANNOTATIONS,
    ANSWERS_GEN_TR,
    ANSWERS_GEN_TARGETS,
    ANSWER_OPTS_GEN_TR,
    ANSWER_OPTS_GEN_TARGETS,
    SOS_TOKEN,
    EOS_TOKEN,
    PAD_INDEX,
    UNK_INDEX
)
from preprocessing.extract_image_features import ExtractImageFeatures


class Visual_Dialog_Dataset(Dataset):
    def __init__(self, json_file_path: str,
                 json_training_file_path: str,
                 images_dir: str,
                 saved_word_to_idx_path: str,
                 params: dict,
                 add_boundary_toks: bool,
                 return_options: bool,
                 dense_annotations_file: str = "",
                 concatenate_history: bool = False,
                 output_path: str = os.getcwd(),
                 save: bool = False,
                 load_saved: bool = True,
                 net_name: str = "VGG",
                 name: str = "VisDiagDataset"
                 ):
        """
        Class constructor.
        :param json_file_path: Path to the json file that will be used to generate the dataset.
        :param json_training_file_path: path the training json file used to generate the vocabulary and the embedding.
        :param images_dir: Path to the directory containing all the images.
        :param saved_vocab_path: Path to the saved vocabulary.
        :param saved_word_to_idx_path: Path to the saved word to index mapping.
        :param dense_annotations_file: Path to the dense annotations file.
        :param path_to_fasttext_model: Path to a pre-trained fasttext bin file.
        :param params: Dict containing all the (hyper) parameters of the experiment.
        :param concatenate_history: If true, the history rounds will be concatenated at each time step
        :param output_path: The path to the directory where the embedding will be stored
        :param save_embedding: If true, the word embedding will be saved
        :param load_saved_vocab: If true a saved vocabulary and embedding will be loaded.
        :param net_name: The name of the feature extractor to be used
        :param name: The name of the dataset
        """
        super(Visual_Dialog_Dataset, self).__init__()
        self.dialog_reader = Reader(json_file_path, images_dir, dense_annotations_file=dense_annotations_file)
        self.mode = self.dialog_reader.mode
        self.name = name + "_" + self.mode
        self.params = params
        self.add_boundary_toks = add_boundary_toks
        self.return_options = return_options
        assert net_name in ["VGG", "Googlenet"]
        if net_name == "VGG":
            self.feature_extactor = VGG_clipped().eval()
        else:
            self.feature_extactor = Googlenet_clipped().eval()
        self.ids = self.dialog_reader.keys()
        if load_saved:
            assert os.path.isfile(saved_word_to_idx_path), "There is no file under the path: {}".format(
                saved_word_to_idx_path)
            with open(saved_word_to_idx_path, "rb") as f:
                self.word_to_idx = pickle.load(f)

        else:
            vocabulary_reader = Vocabulary_Reader([json_training_file_path],
                                                  output_path,
                                                  save=save)
            self.vocabulary = vocabulary_reader.vocabulary
            self.word_to_idx = vocabulary_reader.word_to_idx
        self.concatenate_history = concatenate_history

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx, image_transform=ExtractImageFeatures((224, 224))):
        image_id = self.ids[idx]
        image_path = self.dialog_reader.data_holders[IMAGE_PATHS][image_id]
        caption = self.dialog_reader.data_holders[CAPTIONS][image_id]
        # caption = self.pad_sentence(caption)
        dialog = self.dialog_reader.data_holders[DIALOGS][image_id]

        # Get the indices of each word
        caption = self.from_word_to_idx(caption)

        for i in range(len(dialog)):
            question = self.dialog_reader.data_holders[QUESTIONS][dialog[i][QUESTION]]
            answer = self.dialog_reader.data_holders[ANSWERS][dialog[i][ANSWER]]

            dialog[i][QUESTION] = self.from_word_to_idx(question)
            if self.add_boundary_toks:
                dialog[i][ANSWER] = self.from_word_to_idx([SOS_TOKEN] + answer + [EOS_TOKEN])
            else:
                dialog[i][ANSWER] = self.from_word_to_idx(answer)

            if self.return_options:
                for j in range(len(dialog[i][ANSWER_OPTS])):
                    answer_opt = self.dialog_reader.data_holders[ANSWERS][dialog[i][ANSWER_OPTS][j]]
                    if self.add_boundary_toks:
                        dialog[i][ANSWER_OPTS][j] = self.from_word_to_idx([SOS_TOKEN] + answer_opt + [EOS_TOKEN])
                    else:
                        dialog[i][ANSWER_OPTS][j] = self.from_word_to_idx(answer_opt)

        questions = self._pad_sequences([dialog_round[QUESTION] for dialog_round in dialog])
        history = self._get_history(caption,
                                    [dialog_round[QUESTION] for dialog_round in dialog],
                                    [dialog_round[QUESTION] for dialog_round in dialog],
                                    concatenate_history=self.concatenate_history)

        answers_in = self._pad_sequences([dialog_round[ANSWER][:-1] for dialog_round in dialog])
        answers_out = self._pad_sequences([dialog_round[ANSWER][1:] for dialog_round in dialog])

        item = {IMAGE_ID: torch.tensor(image_id).long(),
                IMG_FEATURES: image_transform(image_path, self.feature_extactor),
                QUESTIONS: questions.long(),
                HISTORY: history.long(),
                ANSWERS_GEN_TR: answers_in.long(),
                ANSWERS_GEN_TARGETS: answers_out.long(),
                }

        if self.return_options:
            if self.add_boundary_toks:
                answer_options_in, answer_options_out = [], []
                for dialog_round in dialog:
                    options = self._pad_sequences([option[:-1] for option in dialog_round[ANSWER_OPTS]])
                    answer_options_in.append(options)

                    options = self._pad_sequences([option[1:] for option in dialog_round[ANSWER_OPTS]])
                    answer_options_out.append(options)

                answer_options_in = torch.stack(answer_options_in, 0)
                answer_options_out = torch.stack(answer_options_out, 0)

                item[ANSWER_OPTS_GEN_TR] = answer_options_in.long()
                item[ANSWER_OPTS_GEN_TARGETS] = answer_options_out.long()
            else:
                answer_options = []
                for dialog_round in dialog:
                    options = self._pad_sequences(dialog_round[ANSWER_OPTS])
                    answer_options.append(options)
                answer_options = torch.stack(answer_options, 0)
                item[ANSWER_OPTS] = answer_options.long()

        if "test" not in self.mode:
            answer_gts = [dialog_round["gt_index"] for dialog_round in dialog]
            item[ANSWER_GT] = torch.tensor(answer_gts).long()

        if self.dialog_reader.mode == "val2018":
            item["round_id"] = torch.tensor(self.dialog_reader.data_holders[DENSE_ANNOTATIONS][image_id]["round_id"])
            item["gt_relevance"] = torch.tensor(
                self.dialog_reader.data_holders[DENSE_ANNOTATIONS][image_id]["gt_relevance"])
        return item, image_path, image_id

    def _pad_sequences(self, sequences):
        for i in range(len(sequences)):
            sequences[i] = sequences[i][: self.params[MAX_SENTENCE_LENGTH]]

        # Pad all sequences to max_sequence_length.
        maxpadded_sequences = torch.full((len(sequences), self.params[MAX_SENTENCE_LENGTH]), fill_value=PAD_INDEX)
        padded_sequences = pad_sequence(
            [torch.tensor(sequence) for sequence in sequences],
            batch_first=True,
            padding_value=PAD_INDEX
        )
        maxpadded_sequences[:, : padded_sequences.size(1)] = padded_sequences
        return maxpadded_sequences

    def _get_history(self, caption, questions, answers, concatenate_history: bool = False):
        # treat caption as a qa-pair --> double the max length
        caption = caption[: self.params[MAX_SENTENCE_LENGTH] * 2]
        history = [caption]
        # Trim the questions and the answers
        for i in range(len(questions)):
            questions[i] = questions[i][: self.params[MAX_SENTENCE_LENGTH]]

        for i in range(len(answers)):
            answers[i] = answers[i][: self.params[MAX_SENTENCE_LENGTH]]

        for (q, a) in zip(questions, answers):
            history.append(q + a)

        # The last q-a pair is not necessary as we do not have 11 dialog rounds.
        history = history[:-1]

        max_length = 2 * self.params[MAX_SENTENCE_LENGTH]

        if concatenate_history:
            max_length = 2 * self.params[MAX_SENTENCE_LENGTH] * len(history)
            concatenated_history = []
            concatenated_history.append(caption)
            for i in range(1, len(history)):
                concatenated_history.append([])
                for j in range(1, i + 1):
                    concatenated_history[i].extend(history[j])

            history = concatenated_history
        padded_history_max = torch.full((len(history), max_length), fill_value=PAD_INDEX)
        padded_history = pad_sequence([torch.tensor(_round) for _round in history], batch_first=True,
                                      padding_value=PAD_INDEX)
        padded_history_max[:, : padded_history.size(1)] = padded_history
        return padded_history_max

    def from_word_to_idx(self, input_string: list):
        """
        Performs a mapping from words to indices.
        :param input_string: the input string after pre-processing, e.g. input_string = [word_1, word_2, ..., word_n]
        :return: list of indices w.r.t. a vocabulary of the words contained in input_string
        """
        assert isinstance(input_string, list)
        input_indices = []
        for word in input_string:
            if word.isdigit():
                word = num2words(word)
            input_indices.append(self.word_to_idx.get(word, UNK_INDEX))
        return input_indices







