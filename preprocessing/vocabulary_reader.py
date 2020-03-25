#!/usr/bin/env python
__author__ = "Mohamed Adnen Abdessaied"
__version__ = "1.0"
__maintainer__ = "Mohamed Adnen Abdessaied"
__email__ = "adnenabdessayed@gmail.com"
__status__ = "Implementation"


"""
This class generates a vocabulary from our data, i.e. from the captions, questions and answers of the training data. 
These data can be found on https://visualdialog.org/data. It also generates an embedding of all vocabulary words using 
the pre-trained english fasttext model (https://fasttext.cc/).   
"""

import os
import pickle
import logging
import argparse
from tqdm import tqdm
from preprocessing.data_readers import Reader
from preprocessing.statics import (
    ANSWERS,
    QUESTIONS,
    CAPTIONS,
    PAD_TOKEN,
    SOS_TOKEN,
    EOS_TOKEN,
    UNK_TOKEN,

    PAD_INDEX,
    SOS_INDEX,
    EOS_INDEX,
    UNK_INDEX
)
# Set the logging level to INFO.
logging.basicConfig(level=logging.INFO)


class Vocabulary_Reader(object):

    def __init__(self, paths_to_data: list,
                 output_path: str,
                 images_dir: str,
                 save: bool = False,
                 name: str = "Vocabulary_Reader"):
        """
        Class constructor.
        :param paths_to_data: The paths to the json files.
        :param output_path: The path to the directory where the embedding will be stored
        :param images_dir: Path to the directory containing the COCO images.
        :param path_to_fasttext_model: path to the fasttext pre-trained model that will be used to generate the
                               embedding.
        :param save_embedding: If true, an embedding of the words will be generated and saved into disk.
        :param name: Name of the vocabulary reader.
        """
        self.name = name
        self.output_path = output_path
        self.images_dir = images_dir
        self.save = save
        self.vocabulary_generated = False
        self.paths_to_data = []
        for path in paths_to_data:
            if not os.path.isfile(path):
                raise FileNotFoundError("There is no file under the path {}".format(path))
            else:
                self.paths_to_data.append(path)
        self.vocabulary_generated = False
        self.embedding_generated = False

        self.vocabulary, self.idx_to_word, self.word_to_idx = self.generate_vocabulary()

    def __del__(self):
        logging.warning("{} successfully deleted...".format(self.name))

    def generate_vocabulary(self):
        """
        A function that generates a dict out of the training data.
        :return: None
        """
        readers = []
        for path in self.paths_to_data:
            readers.append(Reader(path, self.images_dir))

        corpus = [""]  # Initialize the corpus with "" to take care of the padding afterwards.
        for reader in readers:
            for k in [CAPTIONS, ANSWERS, QUESTIONS]:
                progress_bar = tqdm(reader.data_holders[k].values())
                progress_bar.set_description("Loading {} from {}".format(k, reader.name))
                for words in progress_bar:
                    for word in words:
                        corpus += reader.preprocess_word(word)

        # Delete the readers after the corpus is generated.
        for reader in readers:
            del reader

        vocabulary = set(corpus)
        word_to_idx = {}
        word_to_idx[PAD_TOKEN] = PAD_INDEX
        word_to_idx[SOS_TOKEN] = SOS_INDEX
        word_to_idx[EOS_TOKEN] = EOS_INDEX
        word_to_idx[UNK_TOKEN] = UNK_INDEX
        for index, word in enumerate(vocabulary):
            word_to_idx[word] = index + 4

        idx_to_word = {index: word for word, index in word_to_idx.items()}

        self.vocabulary_generated = True
        return vocabulary, idx_to_word, word_to_idx


def main():
    ap = argparse.ArgumentParser()
    # ap.add_argument("-ptr", "--path_training", required=True, help="Path to json training file")
    # ap.add_argument("-pval", "--path_val", required=True, help="Path to json validation file")
    # ap.add_argument("-pte", "--path_test", required=True, help="Path to json test file")
    #ap.add_argument("-img_dir", "--images_dir", required=True, help="Path to the directroy containing the COCO images")
    #
    # ap.add_argument("-o", "--output", required=True, help="path to the directory where the embedding will be saved")
    # ap.add_argument("-f", "--fasttext", required=True, help="Path to pre-trained bin file for fasttext")

    ap.add_argument("-ptr", "--path_training",
                    default="/data/vis_diag/visdial_1.0_train.json",
                    required=False, help="Path to json training file")
    ap.add_argument("-pval", "--path_val",
                    default="/data/vis_diag/visdial_1.0_val.json",
                    required=False, help="Path to json validation file")

    # ap.add_argument("-pte", "--path_test",
    #                 default="/data/vis_diag/visdial_0.9_test.json",
    #                 required=False, help="Path to json test file")
    ap.add_argument("-img_dir", "--images_dir",
                    default="/lhome/mabdess/visual_dialog/data/images",
                    required=False, help="Path to the directroy containing the COCO images")

    ap.add_argument("-o", "--output", required=False,
                    default=".",
                    help="path to the directory where the embedding will be saved")
    #  ap.add_argument("-f", "--fasttext", required=True, help="Path to pre-trained bin file for fasttext")

    args = vars(ap.parse_args())
    paths = [args["path_training"], args["path_val"]] #, args["path_test"]]
    vocabulary_reader = Vocabulary_Reader(
        paths, args["output"], args["images_dir"],
        save=True)

    with open(args["output"] + "/vocab_10.pickle", "wb") as f:
        pickle.dump(vocabulary_reader.vocabulary, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(args["output"] + "/word_to_idx_10.pickle", "wb") as f:
        pickle.dump(vocabulary_reader.word_to_idx, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(args["output"] + "/idx_to_word_10.pickle", "wb") as f:
        pickle.dump(vocabulary_reader.idx_to_word, f, protocol=pickle.HIGHEST_PROTOCOL)

    del vocabulary_reader


if __name__ == "__main__":
    main()
