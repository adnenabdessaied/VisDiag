#!/usr/bin/env python
__author__ = "Mohamed Adnen Abdessaied"
__version__ = "1.0"
__maintainer__ = "Mohamed Adnen Abdessaied"
__email__ = "adnenabdessayed@gmail.com"
__status__ = "Implementation"

"""
We define here some constants to be used in pre-processing the data .
"""

ANSWERS = "answers"
ANSWER = "answer"
PREDICTED_ANSWER = "predicted_answer"
QUESTIONS = "questions"
QUESTION = "question"
CAPTIONS = "captions"
DIALOGS = "dialogs"
DIALOG = "dialog"
DIALOG_LENGTHS = "dialog_lengths"
DIALOG_LENGTH = "dialog_length"
SPLIT = "split"  # specifies the mode: training, validation or test
IMAGE_ID = "image_id"
CAPTION = "caption"  # references a single caption of an image
IMAGE_PATHS = "image_paths"
IMAGE_PATH = "image_path"
MAX_SENTENCE_LENGTH = "max_sentence_length"  # 15
EMBEDDING_DIM = "embedding_size"  # 300
VOCAB_SIZE = "vocab_size"
IMG_FEATURES = "img_features"
HISTORY = "history"
ANSWER_OPTS = "answer_options"
ANSWER_GT = "answer_gt"
DENSE_ANNOTATIONS = "dense_annotations"
ANSWERS_GEN_TR = "answers_gen_tr"
ANSWERS_GEN_TARGETS = "answers_gen_targets"
ANSWER_OPTS_GEN_TR = "answer_opts_gen_tr"
ANSWER_OPTS_GEN_TARGETS = "answer_opts_gen_targets"
DEC_TYPE = "dec_type"

PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<S>"
EOS_TOKEN = "</S>"
UNK_TOKEN = "<UNK>"

PAD_INDEX = 0
SOS_INDEX = 1
EOS_INDEX = 2
UNK_INDEX = 3
