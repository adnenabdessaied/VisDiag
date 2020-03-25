#!/usr/bin/env python
__author__ = "Mohamed Adnen Abdessaied"
__version__ = "1.0"
__maintainer__ = "Mohamed Adnen Abdessaied"
__email__ = "adnenabdessayed@gmail.com"
__status__ = "Implementation"

import cv2
import torch
import copy
import numpy as np
from preprocessing.statics import DIALOGS, ANSWER_OPTS, ANSWERS, QUESTIONS, QUESTION, CAPTIONS
"""
We define here some functions to be used in  tensorboard visualizations.  
"""


def get_tb_image_grid(path_to_image, gts, output, idx, rounds):
    """
    This function randomly takes one image of the batch and displays the gt index and the top predicted indices on top
    of it for each round of the dialog.
    :param path_to_image: The path to the image.
    :param gts: The ground truths of the whole batch.
    :param output: The output of the network for the whole batch.
    :param idx: A randomly chosen index between 0 and batch_size - 1
    :param rounds: A list of rounds we want to visualize.
    :return: A pytorch tensor of decorated images that will be displayed in tensorboard.
    """
    # Read the image
    image = cv2.imread(path_to_image)

    # Convert BGR to RGB as cv2 reads color images in BGR format
    B, R = copy.deepcopy(image[:, :, 0]), copy.deepcopy(image[:, :, -1])
    image[:, :, 0], image[:, :, -1] = R, B

    # Resize every image to a fixed size (600, 400)
    image = cv2.resize(image, (600, 400))

    # Make a copy of the each for each dialog round
    images = [copy.deepcopy(image) for _ in range(len(rounds))]

    # Extract the gts corresponding to the randomly chosen image
    gts = gts[idx * 10: (idx + 1) * 10]
    gts = gts[rounds]
    gts_text = [str(gt.item()) for gt in gts]

    output = output.detach()

    # Extract the outputs corresponding to the randomly chosen image
    output = output[idx * 10: (idx + 1) * 10][:]

    output = output[rounds][:]

    _, output_idx = output.sort(dim=1, descending=True)
    output_size = output.size()

    top_1 = [output_idx[i][:1] for i in range(output_size[0])]
    top_1_texts = ["[{}]".format(str(predicted.item())) for predicted in top_1]

    top_3 = [output_idx[i][:3].tolist() for i in range(output_size[0])]
    top_3_texts = ["[{}, {}, {}]".format(*predicted) for predicted in top_3]

    top_5 = [output_idx[i][:5].tolist() for i in range(output_size[0])]
    top_5_texts = ["[{}, {}, {}, {}, {}]".format(*predicted) for predicted in top_5]

    image_grid = []

    # Decorate each image as described in function doc string.
    for i, (img, gt_text, top_1_text, top_3_text, top_5_text) in enumerate(zip(
            images, gts_text, top_1_texts, top_3_texts, top_5_texts)):

        cv2.putText(img, "Round {}:".format(rounds[i] + 1), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(img, "gt index:" + gt_text, (400, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, "top_1_pred:" + top_1_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(img, "top_3_pred:" + top_3_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(img, "top_5_pred:" + top_5_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        image_grid.append(torch.from_numpy(img)/255.0)

    image_grid = torch.stack(image_grid, dim=0)

    return image_grid


def decorate_tb_image(path_to_image, gts, output, idx, best_k_pred, image_id, dialog_reader):
    """

    :param path_to_image:
    :param gts:
    :param output:
    :param idx:
    :param best_k_pred:
    :param image_id:
    :param dialog_reader:
    :return:
    """
    # Read the image
    image = cv2.imread(path_to_image)

    # Convert BGR to RGB as cv2 reads color images in BGR format
    B, R = copy.deepcopy(image[:, :, 0]), copy.deepcopy(image[:, :, -1])
    image[:, :, 0], image[:, :, -1] = R, B

    # Resize every image to a fixed size (600, 400)
    image = cv2.resize(image, (1000, 1000))

    # Extract the gts corresponding to the randomly chosen image
    gts = gts[idx * 10: (idx + 1) * 10]
    dialog = dialog_reader.data_holders[DIALOGS][image_id]

    raw_questions = []
    raw_answers = []
    raw_predicted_answers = []

    caption = dialog_reader.data_holders[CAPTIONS][image_id]
    caption = " ".join(caption)
    cv2.putText(image, "C: {}".format(caption), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Detach tensor from graph to avoid memory leaks
    output = output.detach()

    # Extract the outputs corresponding to the randomly chosen image
    output = output[idx * 10: (idx + 1) * 10][:]
    _, output_indices = output.sort(dim=1, descending=True)
    for (diag_round, gt_idx, predicted_idx) in zip(dialog, gts.tolist(), output_indices.tolist()):
        raw_questions.append(" ".join(dialog_reader.data_holders[QUESTIONS][diag_round[QUESTION]]) + "?")
        raw_answers.append(" ".join(dialog_reader.data_holders[ANSWERS][diag_round[ANSWER_OPTS][gt_idx]]))
        raw_predicted_answers.append(" ".join(dialog_reader.data_holders[ANSWERS][
                                                  diag_round[ANSWER_OPTS][predicted_idx[best_k_pred - 1]]]))

    raw_text = []
    for raw_question, raw_predicted_answer, raw_answer in zip(raw_questions, raw_predicted_answers, raw_answers):
        raw_text.append(raw_question)
        raw_text.append(raw_predicted_answer)
        raw_text.append(raw_answer)

    # Add a black background where the dialog will be displayed
    black_background = np.zeros((1000, 1000, 3), dtype=np.uint8)
    for i, text in enumerate(raw_text):
        if i % 3 == 0:
            color = (255, 255, 255)
        elif (i - 1) % 3 == 0:
            color = (0, 255, 0)
        else:
            color = (255, 0, 0)
        cv2.putText(black_background, text, (10, 31 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    decorated_image = np.concatenate((image, black_background), axis=1)
    return decorated_image / 255.0


def decorate_tb_image_with_gen_answers(path_to_image, gts, gen_ansers, idx, image_id, dialog_reader):
    """

    :param path_to_image:
    :param gts:
    :param gen_ansers:
    :param idx:
    :param image_id:
    :param dialog_reader:
    :return:
    """
    # Read the image
    image = cv2.imread(path_to_image)

    # Convert BGR to RGB as cv2 reads color images in BGR format
    B, R = copy.deepcopy(image[:, :, 0]), copy.deepcopy(image[:, :, -1])
    image[:, :, 0], image[:, :, -1] = R, B

    # Resize every image to a fixed size (600, 400)
    image = cv2.resize(image, (1000, 1000))

    # Extract the gts corresponding to the randomly chosen image
    gts = gts[idx * 10: (idx + 1) * 10]
    dialog = dialog_reader.data_holders[DIALOGS][image_id]

    raw_questions = []
    raw_answers = []

    caption = dialog_reader.data_holders[CAPTIONS][image_id]
    caption = " ".join(caption)
    cv2.putText(image, "C: {}".format(caption), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Extract the generated answers corresponding to the randomly chosen image
    raw_predicted_answers = gen_ansers[idx]

    for (diag_round, gt_idx) in zip(dialog, gts.tolist()):
        raw_questions.append(" ".join(dialog_reader.data_holders[QUESTIONS][diag_round[QUESTION]]) + "?")
        raw_answers.append(" ".join(dialog_reader.data_holders[ANSWERS][diag_round[ANSWER_OPTS][gt_idx]]))

    raw_text = []
    for raw_question, raw_predicted_answer, raw_answer in zip(raw_questions, raw_predicted_answers, raw_answers):
        raw_text.append(raw_question)
        raw_text.append(raw_predicted_answer)
        raw_text.append(raw_answer)

    # Add a black background where the dialog will be displayed
    black_background = np.zeros((1000, 1000, 3), dtype=np.uint8)
    for i, text in enumerate(raw_text):
        if i % 3 == 0:
            color = (255, 255, 255)
        elif (i - 1) % 3 == 0:
            color = (0, 255, 0)
        else:
            color = (255, 0, 0)
        cv2.putText(black_background, text, (10, 31 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    decorated_image = np.concatenate((image, black_background), axis=1)
    return decorated_image / 255.0


def decorate_tb_image_with_gen_answers_test(path_to_image, gen_ansers, idx, image_id, dialog_reader):
    """

    :param path_to_image:
    :param gts:
    :param gen_ansers:
    :param idx:
    :param image_id:
    :param dialog_reader:
    :return:
    """
    # Read the image
    image = cv2.imread(path_to_image)

    # Convert BGR to RGB as cv2 reads color images in BGR format
    B, R = copy.deepcopy(image[:, :, 0]), copy.deepcopy(image[:, :, -1])
    image[:, :, 0], image[:, :, -1] = R, B

    # Resize every image to a fixed size (600, 400)
    image = cv2.resize(image, (1000, 1000))

    # Extract the gts corresponding to the randomly chosen image
    dialog = dialog_reader.data_holders[DIALOGS].get(image_id)

    raw_questions = []

    for diag_round in dialog:
        raw_questions.append(" ".join(dialog_reader.data_holders[QUESTIONS][diag_round[QUESTION]]) + "?")

    caption = dialog_reader.data_holders[CAPTIONS][image_id]
    caption = " ".join(caption)
    cv2.putText(image, "C: {}".format(caption), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Extract the generated answers corresponding to the randomly chosen image
    raw_predicted_answers = gen_ansers[idx]

    raw_text = []
    for raw_question, raw_predicted_answer in zip(raw_questions, raw_predicted_answers):
        raw_text.append(raw_question)
        raw_text.append(raw_predicted_answer)

    # Add a black background where the dialog will be displayed
    black_background = np.zeros((1000, 1000, 3), dtype=np.uint8)
    for i, text in enumerate(raw_text):
        if i % 2 == 0:
            color = (255, 255, 255)
        else:
            color = (0, 255, 0)

        cv2.putText(black_background, text, (10, 31 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    decorated_image = np.concatenate((image, black_background), axis=1)
    return decorated_image / 255.0

