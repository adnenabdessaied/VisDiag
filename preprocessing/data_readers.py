#!/usr/bin/env python
__author__ = "Mohamed Adnen Abdessaied"
__version__ = "1.0"
__maintainer__ = "Mohamed Adnen Abdessaied"
__email__ = "adnenabdessayed@gmail.com"
__status__ = "Implementation"

"""
The data readers implemented here are supposed to read all the data related to a specific image_id (i.e. answers,
questions, dialogs, captions, images) from disk. The data can be accessed using the json files provided on
https://visualdialog.org/data.

Each data reader has to implement the following methods: 
    - "__len__" :returns the length of reader specific data 
    - "__getitem__" :returns the data related to the specified image_id
    - "__keys__" :returns a list of feasible image_ids the reader can use to generate data.
"""

import ijson
import json
import os
import cv2
import logging

from nltk.tokenize import word_tokenize

import string
from num2words import num2words
from preprocessing.contractions import CONTRACTIONS_DICT

from preprocessing.statics import (
    ANSWERS,
    ANSWER,
    QUESTIONS,
    QUESTION,
    CAPTIONS,
    DIALOGS,
    DIALOG,
    DIALOG_LENGTHS,
    DIALOG_LENGTH,
    IMAGE_ID,
    CAPTION,
    IMAGE_PATHS,
    IMAGE_PATH,
    DENSE_ANNOTATIONS
)
logging.basicConfig(level=logging.INFO)


class Reader(object):
    """
    A class that reads the answers, questions and dialogs based on a specific json file.
    """
    def __init__(self, json_file_path: str, images_dir: str, name: str = "Dialog_Reader",
                 dense_annotations_file: str = ""):
        """
        Class constructor.
        :param json_file_path: The path to the json file containing the dialog data.
        :param images_dir: Path to the directory containing the COCO images.
        :param name: Name of the reader.
        :param dense_annotations_file: The dense annotations of the validation data.
        """
        self.json_file_path = json_file_path
        self.images_dir = images_dir
        # Initialize the data holder of this reader and store them in a dict.
        # The answers and questions data holders are straightforward. They uses the same order of the answers and
        # questions to store them in the dict. The captions however use the image_id as keys and the actual captions
        # as values.
        self.data_holders = {ANSWERS: {},
                             QUESTIONS: {},
                             CAPTIONS: {},
                             DIALOGS: {},
                             DIALOG_LENGTHS: {},
                             IMAGE_PATHS: {}
                             }

        # Check if the data is used for training, validation or test
        with open(json_file_path, "r") as f:
            items = ijson.items(f, "split")
            for item in items:
                self.mode = item
        self.name = name + "_" + self.mode
        # Store the data in the specific holder
        self.read_from_huge_json(ANSWERS)
        self.read_from_huge_json(QUESTIONS)

        # Read the dense annotations for the validation data
        if self.mode == "val2018":
            self.data_holders[DENSE_ANNOTATIONS] = self.read_dense_annotations()

        # Add the empty answer and question using key = -1 to make padding afterwards easier.
        self.data_holders[ANSWERS][-1] = [""]
        self.data_holders[QUESTIONS][-1] = [""]

        self.read_from_huge_json(DIALOGS)
        # self.read_image_captions_and_paths()
        self.num_answers = len(self.data_holders[ANSWERS])
        self.num_questions = len(self.data_holders[QUESTIONS])
        self.num_dialogs = len(self.data_holders[DIALOGS])
        self.num_images = len(self.data_holders[CAPTIONS])

    def __del__(self):
        logging.warning("{} successfully deleted ...".format(self.name))

    def read_from_huge_json(self, data_to_be_read: str, pad_dialogs: bool = True):
        """
        A method that reads a huge json file as a stream and extracts specific data in order to avoid memory issues.
        :param data_to_be_read: The specific data that has to be read (e.g. "answers", "questions" or dialogs).
        :param pad_dialogs: If the true, the dialogs will be padded since the test data doesn't always have 10
                            dialog rounds.
        :return: None.
        """
        with open(self.json_file_path, "r") as f:
            items = ijson.items(f, "data." + data_to_be_read + ".item")
            for i, item in enumerate(items):
                if data_to_be_read == DIALOGS:
                    dialog = item[DIALOG]
                    for i in range(len(dialog)):
                        if ANSWER not in dialog[i]:
                            dialog[i][ANSWER] = -1
                    if pad_dialogs:
                        for _ in range(10 - len(dialog)):
                            dialog.append({QUESTION: -1, ANSWER: -1})

                    self.data_holders[data_to_be_read][item[IMAGE_ID]] = dialog
                    self.data_holders[DIALOG_LENGTHS][item[IMAGE_ID]] = len(item[DIALOG])
                    self.data_holders[CAPTIONS][item[IMAGE_ID]] = self.preprocess_string(item[CAPTION])
                    self.data_holders[IMAGE_PATHS][item[IMAGE_ID]] = self.find_image_path(item[IMAGE_ID])

                else:
                    self.data_holders[data_to_be_read][i] = self.preprocess_string(item)
            if data_to_be_read == DIALOGS:
                logging.info("Done reading dialogs and captions of {} ...".format(self.mode))
            else:
                logging.info("Done reading " + data_to_be_read + " of {} ...".format(self.mode))

    def read_dense_annotations(self, dense_annotations_file="/data/vis_diag/visdial_1.0_val_dense_annotations.json"):
        """
        Reads the dense annotations of the validation data to able to compute the NDCG metric. The dense annotations
        have been introduced in v1.0.
        :return: None
        """
        with open(dense_annotations_file, "rb") as f:
            dense_annotations_ = json.load(f)
        dense_annotations = {}
        for annotation in dense_annotations_:
            dense_annotations[annotation[IMAGE_ID]] = {"round_id": annotation["round_id"],
                                                       "gt_relevance": annotation["gt_relevance"]}
        return dense_annotations

    def print_dialog(self, image_id: int, display_image: bool = False):
        """
        A method that prints a dialog to the screen and shows the image used to generate the dialog.
        It helps to check the consistency of the data loading.
        :param image_id: The index of the image used to generate the dialog.
        :param display_image: If true, the image on which the dialog was based will be displayed as well.
        :return: None.
        """
        assert_condition = len(self.data_holders[ANSWERS]) > 0 and len(self.data_holders[QUESTIONS]) > 0 and len(
            self.data_holders[CAPTIONS]) > 0
        assert assert_condition, "Load the data first!"
        dialog = self.data_holders[DIALOGS][image_id]
        caption = " ".join(self.data_holders[CAPTIONS][image_id])
        print(CAPTION + ": {}".format(caption))
        for i, dialog_round in enumerate(dialog):
            question = " ".join(self.data_holders[QUESTIONS][dialog_round[QUESTION]])
            try:
                answer = " ".join(self.data_holders[ANSWERS][dialog_round[ANSWER]])
            except KeyError:
                answer = "?"
            print("Q {}: {}".format(i, question))
            print("A {}: {}".format(i, answer))
        if display_image:
            image = cv2.imread(self.data_holders[IMAGE_PATHS][image_id], 3)
            cv2.imshow(caption, image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def image_name(self, image_id: int):
        """
        Reconstructs the real name of the image based on an image_id.
        :param image_id: the image id.
        :return: The real name of the image.
        """
        image_id_expanded = "0" * (12 - len(str(image_id))) + str(image_id)
        if self.mode == "train":
            return "COCO_train2014_" + image_id_expanded + ".jpg", "COCO_val2014_" + image_id_expanded + ".jpg"
        elif "2018" in self.mode:
            return "VisualDialog_" + self.mode + "_" + image_id_expanded + ".jpg"
        elif "2014" in self.mode:
            return "COCO_" + self.mode + "_" + image_id_expanded + ".jpg"
        else:
            raise FileNotFoundError

    def find_image_path(self, image_id: int):
        """
        Returns the path of the image based on the image id.
        :param image_id: the image id.
        :return The absolute path to the image.
        """
        if self.mode == "train":
            for image_name in self.image_name(image_id):
                for folder in ["train2014", "val2014"]:
                    folder_path = os.path.join(self.images_dir, folder)
                    image_path = os.path.join(folder_path, image_name)
                    if os.path.isfile(image_path):
                        return image_path

        else:
            if "2018" in self.mode:
                folder = "VisualDialog_" + self.mode
            else:
                folder = self.mode
            folder_path = os.path.join(self.images_dir, folder)
            image_path = os.path.join(folder_path, self.image_name(image_id))
            return image_path

    def __len__(self):
        return self.num_dialogs

    def __getitem__(self, image_id):
        return {
            IMAGE_ID: image_id,
            IMAGE_PATH: self.data_holders[IMAGE_PATHS][image_id],
            CAPTION: self.data_holders[CAPTIONS][image_id],
            DIALOGS: self.data_holders[DIALOGS][image_id],
            DIALOG_LENGTH: self.data_holders[DIALOG_LENGTHS][image_id]
        }

    def keys(self):
        return list(self.data_holders[DIALOGS].keys())

    @staticmethod
    def preprocess_word(word):
        punc = string.punctuation
        punc = "".join(c for c in punc if c is not "\'")
        word = "".join([c for c in word if c not in punc])
        if word.isdigit():
            word = num2words(word)
        elif word in CONTRACTIONS_DICT:
            word = CONTRACTIONS_DICT[word]

        return word_tokenize(word)

    @staticmethod
    def preprocess_string(input_string: str):
        """
        Method that pre-processes an input string:
          lower-casing -> replacing numbers with letters + removing contractions + removing punctuations -> tokenizing
          -> removing stop words.
        :param input_string: The string to be pre-processed.
        :return: (list): tokenized, pre-processed string.
    """
        punc = string.punctuation
        punc = "".join(c for c in punc if c is not "\'")
        input_string = input_string.replace("-", " ")
        input_string = "".join([c for c in input_string if c not in punc])
        input_string = input_string.lower()
        input_string = input_string.split(" ")
        input_string = [CONTRACTIONS_DICT[word] if word in CONTRACTIONS_DICT else word for word in input_string]
        input_string = list(map(lambda x: x.replace("\'", " "), input_string))

        for i, word in enumerate(input_string):
            input_string[i] = "".join([c for c in word if c not in punc])
        input_string = word_tokenize(" ".join(input_string))

        for i, word in enumerate(input_string):
            if word.isdigit():
                num_str = num2words(word)
                num_str = "".join([c for c in num_str if c not in punc])
                num_str = word_tokenize(num_str)
                input_string[i:i] = num_str
                input_string.remove(word)
        return input_string
