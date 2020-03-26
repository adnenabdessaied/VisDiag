#!/usr/bin/env python
__author__ = "Mohamed Adnen Abdessaied"
__version__ = "1.0"
__maintainer__ = "Mohamed Adnen Abdessaied"
__email__ = "adnenabdessayed@gmail.com"
__status__ = "Implementation"

import torch
import argparse
import logging
import json
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
from datetime import datetime
from tqdm import tqdm

from torch.utils.data import DataLoader
import torch.nn as nn
from preprocessing.interactive_dataset_inference import Interactive_Dataset_Inference
from preprocessing.interactive_dataset_training import Interactive_Dataset_Training
from preprocessing.statics import ANSWER, QUESTION, DIALOG, CAPTION, IMAGE_PATH, QUESTIONS, PREDICTED_ANSWER
from preprocessing.statics import (
    EMBEDDING_DIM,
    MAX_SENTENCE_LENGTH,
    DEC_TYPE,
    ANSWERS_GEN_TARGETS
)

logging.basicConfig(level=logging.INFO)


def _get_current_timestamp() -> str:
    """
    A function that returns the current timestamp to be used to name the checkpoint files of the model.
    :return: The current timestamp.
    """
    current_time = datetime.now()
    current_time = current_time.strftime("%d-%b-%Y (%H:%M:%S.%f)")
    return current_time


def _get_args():
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("-w2i",
                            "--word_to_index",
                            required=False,
                            default="../word_to_idx.pickle",
                            help="Path to a saved word to index mapping.")

    arg_parser.add_argument("-i2w",
                            "--index_to_word",
                            required=False,
                            default="../idx_to_word.pickle",
                            help="Path to a saved index to word mapping.")

    arg_parser.add_argument("-b",
                            "--batch_size",
                            required=False,
                            default=1,
                            help="Batch size.")

    arg_parser.add_argument("-embdim",
                            "--embedding_dim",
                            required=False,
                            default=300,
                            help="Dimension of the word embedding.")

    arg_parser.add_argument("-m",
                            "--max_len",
                            required=False,
                            default=10,
                            help="Maximum sentence length.")

    arg_parser.add_argument("-o",
                            "--output",
                            required=False,
                            default="../output_interactive",
                            help="Folder where the fine tuned network will be saved")
    args = vars(arg_parser.parse_args())
    return args


def _freeze_weights(layer):
    """
    This function freezes the parameters of a layer
    :param layer: The layer whose parameters we want to freeze
    :return: None
    """
    # https://stackoverflow.com/questions/11637293/iterate-over-object-attributes-in-python
    # -->
    attributes = [getattr(layer, a) for a in dir(layer) if not a.startswith('__') and not callable(getattr(layer, a))]
    # <--
    for a in attributes:
        if isinstance(a, nn.Parameter):
            a.requires_grad = False


class GUI(object):
    """
    This is the implementation of a gui that will be used for our interactive mode.
    """
    # Define class variables that hold the data generated whilst using the interactive mode.
    network_loaded = False
    images_paths = []
    data = {}
    questions = []
    answers = []
    predicted_answers = []
    captions = []
    net, optimizer = None, None
    args = _get_args()
    params = {EMBEDDING_DIM: args["embedding_dim"],
              MAX_SENTENCE_LENGTH: args["max_len"],
              DEC_TYPE: "gen"
              }
    # Use GPU if available
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        logging.info("Using {}".format(torch.cuda.get_device_name()))
    else:
        device = torch.device("cpu")
        logging.info("Using the CPU")

    def __init__(self, height=800, width=1800):
        """
        Class constructor.
        :param height: The height of the tk canvas.
        :param width: The width of the tk canvas
        """
        super(object, self).__init__()
        self.root_window = tk.Tk()
        self.image_imported = False
        self.dialog_saved = False
        canvas = tk.Canvas(self.root_window, height=height, width=width)
        canvas.pack()
        self.question_num = 0
        self.answer_num = 0
        self.prediction = 0
        self.answer = ""
        self.caption_stored = False
        greeting_txt = tk.StringVar()
        greeting_txt.set("Welcome! You are now using the interactive mode of our visual dialog system.")
        greeting_msg = tk.Message(self.root_window, textvariable=greeting_txt, aspect=10000)

        import_image_button = tk.Button(self.root_window, text="import image", command=self.load_img)
        ask_question_button = tk.Button(self.root_window, text="ask question", command=self.ask_question)
        get_answer_button = tk.Button(self.root_window, text="get answer", command=self.print_answer)
        give_feedback_button = tk.Button(self.root_window, text="give feedback", command=self.give_feedback)
        save_dialog_button = tk.Button(self.root_window, text="save dialog", command=self.save_dialog)
        next_button = tk.Button(self.root_window, text="next", command=self.next)
        train_button = tk.Button(self.root_window, text="train", command=self.train, bg="#af290c")
        save_net_button = tk.Button(self.root_window, text="save network", command=self.save_network, bg="#63bf06")
        save_to_json_button = tk.Button(self.root_window, text="save dialogs to json",
                                        command=self.save_dialogs_to_json)
        load_net_button = tk.Button(self.root_window, text="load network", command=self.load_network)

        greeting_msg.place(x=630, y=5)
        import_image_button.place(x=15, y=30)
        ask_question_button.place(x=150, y=30)
        get_answer_button.place(x=280, y=30)
        give_feedback_button.place(x=400, y=30)
        save_dialog_button.place(x=540, y=30)
        train_button.place(x=665, y=30)
        save_net_button.place(x=745, y=30)
        next_button.place(x=1715, y=20)
        save_to_json_button.place(x=1535, y=20)
        load_net_button.place(x=1400, y=20)

    def load_img(self):
        """
        Loads an image to the gui.
        :return: None.
        """
        try:
            if GUI.network_loaded:
                img_path = filedialog.askopenfilename(title='open')
                GUI.images_paths.append(img_path)
                image = Image.open(img_path)
                image = ImageTk.PhotoImage(image.resize((670, 700), Image.ANTIALIAS))
                image_panel = tk.Label(self.root_window, image=image)
                image_panel.image = image
                image_panel.place(x=1100, y=60)
                self.image_imported = True
                self.clear_error_msg_bottom()
                self.clear_msg_top()
                give_caption_button = tk.Button(self.root_window, text="give caption", command=self.get_caption)
                give_caption_button.place(x=1100, y=765)
            else:
                self.clear_error_msg_bottom()
                error_txt = tk.StringVar()
                error_txt.set("You have to load a pre-trained network first.")
                error_msg = tk.Message(self.root_window, textvariable=error_txt, aspect=10000, fg="#af290c")
                error_msg.place(x=25, y=600)
        except:
            del GUI.images_paths[-1]

    def load_network(self):
        """
        Loads an image to the gui.
        :return: None.
        """
        if not GUI.network_loaded:
            network_path = filedialog.askopenfilename(title='open')
            most_recent_chkpt = torch.load(network_path, map_location=lambda storage, loc: storage.cuda(0))
            GUI.net = most_recent_chkpt["net"]
            GUI.net.load_state_dict(most_recent_chkpt["net_state_dict"])

            # We want to train further
            GUI.net.train()

            # Send the net first to the device to avoid potential runtime errors caused by the optimizer if we resume
            # training on a different device
            GUI.net.to(GUI.device)

            GUI.optimizer = most_recent_chkpt["optimizer"]
            GUI.optimizer.load_state_dict(most_recent_chkpt["optimizer_state_dict"])
            children = [list(subnet.children()) for subnet in list(GUI.net.children())]
            for child in children:
                for layer in child:
                    _freeze_weights(layer)
            # Unfreeze the parameters of the last layer, i.e. the fc layer of the decoder.
            children[-1][2].bias.requires_grad = True
            children[-1][2].weight.requires_grad = True

            # update the optimizer to fine tune the last layer only
            GUI.optimizer.param_groups[0]["params"] = list(filter(lambda p: p.requires_grad, GUI.net.parameters()))
            GUI.network_loaded = True
            self.clear_error_msg_bottom()
            msg_txt = tk.StringVar()
            msg_txt.set("[INFO] Pre-trained model successfully loaded ...")
            msg = tk.Message(self.root_window, textvariable=msg_txt, aspect=10000, fg="#af290c")
            msg.place(x=880, y=33)

    def get_caption(self):
        def save(enter):
            if not self.caption_stored:
                GUI.captions.append(caption_field.get())
                self.caption_stored = True
                self.clear_error_msg_bottom()

        caption_field = tk.Entry(self.root_window, bd=1)
        caption_field.bind("<Return>", save)
        caption_field.place(x=1230, y=770)

    def ask_question(self):
        """
        Adds a new field where a new question can be asked.
        :return: None.
        """
        def save(enter):
            if len(GUI.questions) == self.question_num - 1:
                GUI.questions.append(question_field.get())

        if self.question_num < 10 and self.image_imported and self.question_num == self.answer_num and \
                self.caption_stored:
            self.clear_error_msg_bottom()
            self.clear_msg_top()
            self.question_num += 1
            question_label = tk.Label(self.root_window, text="Question {}".format(self.question_num))
            question_field = tk.Entry(self.root_window, bd=1)
            question_field.bind("<Return>", save)
            question_label.place(x=25, y=50 * (self.question_num + 1))
            question_field.place(x=130, y=50 * (self.question_num + 1))
            self.dialog_saved = False
        elif not self.image_imported:
            self.clear_error_msg_bottom()
            error_txt = tk.StringVar()
            error_txt.set("Sorry! You have to import an image first.")
            error_msg = tk.Message(self.root_window, textvariable=error_txt, aspect=10000, fg="#af290c")
            error_msg.place(x=25, y=600)
        elif not self.caption_stored:
            self.clear_error_msg_bottom()
            error_txt = tk.StringVar()
            error_txt.set("Sorry! You have to caption the image first.")
            error_msg = tk.Message(self.root_window, textvariable=error_txt, aspect=10000, fg="#af290c")
            error_msg.place(x=25, y=600)
        elif self.question_num >= 10:
            self.clear_error_msg_bottom()

            error_txt = tk.StringVar()
            error_txt.set("Sorry! You cannot exceed 10 questions.")
            error_msg = tk.Message(self.root_window, textvariable=error_txt, aspect=10000, fg="#af290c")
            error_msg.place(x=25, y=600)

    def print_answer(self):
        """
        Prints the predicted answer from our generative visual dialog to the gui.
        :return: None
        """
        if self.question_num == self.prediction + 1 and GUI.network_loaded:
            self.prediction += 1
            answer_label = tk.Label(self.root_window, text="Predicted answer:")
            predicted_answer_var = tk.StringVar()
            dialog = []
            for i in range(len(GUI.questions)-1):
                dialog_round = {QUESTION: GUI.questions[i],
                                ANSWER: GUI.answers[i]}
                dialog.append(dialog_round)
            # Add the last dialog round and append the missing rounds
            dialog.append({QUESTION: GUI.questions[-1],
                           ANSWER: ""})
            for _ in range(10 - len(dialog)):
                dialog.append({QUESTION: "", ANSWER: ""})

            data_dict = {IMAGE_PATH: GUI.images_paths[-1],
                         CAPTION: GUI.captions[-1],
                         DIALOG: dialog,
                         QUESTIONS: GUI.questions}
            inference_dataset = Interactive_Dataset_Inference(data_dict, GUI.params, GUI.args["word_to_index"])
            inference_dataloader = DataLoader(inference_dataset)
            net = GUI.net
            net.eval()
            device = GUI.device
            for batch, _ in inference_dataloader:
                batch = dict(zip(batch.keys(), map(lambda x: x.to(device), batch.values())))
                with torch.no_grad():
                    _, gen_output = net(batch)
                self.set_answer(gen_output[0][self.question_num - 1])
                GUI.predicted_answers.append(self.answer)
            predicted_answer_var.set(self.answer)
            predicted_msg = tk.Message(self.root_window, textvariable=predicted_answer_var, aspect=10000, fg="#498e04")
            answer_label.place(x=320, y=50 * (self.question_num + 1))
            predicted_msg.place(x=440, y=50 * (self.question_num + 1))
        elif not GUI.network_loaded:
            self.clear_error_msg_bottom()
            error_txt = tk.StringVar()
            error_txt.set("You have to load a pre-trained network first.")
            error_msg = tk.Message(self.root_window, textvariable=error_txt, aspect=10000, fg="#af290c")
            error_msg.place(x=25, y=600)

    def set_answer(self, answer):
        """
        A setter function for the "answer" attribute.
        :param answer: The predicted answer.
        :return: None.
        """
        self.answer = answer

    def give_feedback(self):
        """
        Adds a new field where the user feedback can be input.
        :return: None.
        """
        def save(enter):
            if len(GUI.answers) == self.answer_num - 1:
                GUI.answers.append(feedback_field.get())
        if self.answer_num == self.question_num - 1:
            self.answer_num += 1
            feedback_label = tk.Label(self.root_window, text="Feedback:")
            feedback_field = tk.Entry(self.root_window, bd=1)
            feedback_field.bind("<Return>", save)
            feedback_label.place(x=730, y=50 * (self.answer_num + 1))
            feedback_field.place(x=800, y=50 * (self.answer_num + 1))

    def save_dialog(self):
        if len(GUI.questions) == len(GUI.answers) and len(GUI.answers) != 0 and not self.dialog_saved:
            self.clear_error_msg_bottom()

            dialog = []
            for i in range(len(GUI.questions)):
                dialog_round = {QUESTION: GUI.questions[i],
                                PREDICTED_ANSWER: GUI.predicted_answers[i],
                                ANSWER: GUI.answers[i]}
                dialog.append(dialog_round)
            GUI.questions = []
            GUI.answers = []
            GUI.predicted_answers = []
            data_dict = {DIALOG: dialog,
                         CAPTION: GUI.captions[-1]}
            GUI.data[GUI.images_paths[-1]] = data_dict
            self.dialog_saved = True
        elif len(GUI.questions) != len(GUI.answers):
            self.clear_error_msg_bottom()

            error_txt = tk.StringVar()
            error_txt.set("Numbers of questions and answers do not match.")
            error_msg = tk.Message(self.root_window, textvariable=error_txt, aspect=10000, fg="#af290c")
            error_msg.place(x=25, y=600)
        elif len(GUI.answers) == 0 and not self.dialog_saved:
            self.clear_error_msg_bottom()

            error_txt = tk.StringVar()
            error_txt.set("You need to enter at least one question-answer pair.")
            error_msg = tk.Message(self.root_window, textvariable=error_txt, aspect=10000, fg="#af290c")
            error_msg.place(x=25, y=600)
        else:
            self.clear_error_msg_bottom()

            error_txt = tk.StringVar()
            error_txt.set("Dialog is already saved.")
            error_msg = tk.Message(self.root_window, textvariable=error_txt, aspect=10000, fg="#af290c")
            error_msg.place(x=25, y=600)

    def train(self):
        if self.dialog_saved and GUI.network_loaded:
            tr_dataset = Interactive_Dataset_Training(GUI.data, GUI.params, GUI.args["word_to_index"])
            logging.info("Data successfully loaded ...")
            batch_size = min(GUI.args["batch_size"], len(tr_dataset))
            logging.info("Constructing the data loaders ...")

            tr_data_loader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=batch_size != 1, num_workers=1)
            pbar_train = tqdm(tr_data_loader)
            pbar_train.set_description("Fine tuning")

            net = GUI.net
            net.train()
            device = GUI.device
            optimizer = GUI.optimizer
            criterion = nn.CrossEntropyLoss()

            for batch, image_paths in pbar_train:
                # Send the data to the appropriate device
                batch = dict(zip(batch.keys(), map(lambda x: x.to(device), batch.values())))

                optimizer.zero_grad()
                output = net(batch)

                # Shape: (batch_size * rounds, answer_opts)
                output = output.view(-1, output.size(-1))

                # Compute the training loss
                tr_loss = criterion(output, batch[ANSWERS_GEN_TARGETS].view(-1))

                # Back propagation with anomaly detection -> Makes it easier to locate the faulty parts of the net
                # if some undesirable phenomena happen, e.g. if some layers produce NaN of Inf values.
                with torch.autograd.detect_anomaly():
                    tr_loss.backward()

                # Clamp the gradients to avoid explosion
                nn.utils.clip_grad_norm_(net.parameters(), 5)

                optimizer.step()

                # Release GPU memory cache
                torch.cuda.empty_cache()

        elif not self.dialog_saved:
            self.clear_error_msg_bottom()
            error_txt = tk.StringVar()
            error_txt.set("You need to save the dialog before proceeding.")
            error_msg = tk.Message(self.root_window, textvariable=error_txt, aspect=10000, fg="#af290c")
            error_msg.place(x=25, y=600)
        elif not GUI.network_loaded:
            self.clear_error_msg_bottom()
            error_txt = tk.StringVar()
            error_txt.set("You have to load a pre-trained network first.")
            error_msg = tk.Message(self.root_window, textvariable=error_txt, aspect=10000, fg="#af290c")
            error_msg.place(x=25, y=600)

    def save_network(self):
        args = _get_args()
        timestamp = _get_current_timestamp()
        torch.save({
            "net": GUI.net,
            "net_state_dict": GUI.net.state_dict(),
            "optimizer": GUI.optimizer,
            "optimizer_state_dict": GUI.optimizer.state_dict(),
        }, os.path.join(args["output"], "checkpoint_{}.pth".format(timestamp)))
        self.clear_error_msg_bottom()
        self.clear_msg_top()
        msg_txt = tk.StringVar()
        msg_txt.set("[INFO] Model successfully saved ...")
        msg = tk.Message(self.root_window, textvariable=msg_txt, aspect=10000, fg="#af290c")
        msg.place(x=880, y=33)

    @staticmethod
    def save_dialogs_to_json():
        path_json = os.path.join(GUI.args["output"], "logged_dialogs.json")
        with open(path_json, "w") as f:
            json.dump(GUI.data, f, indent=4)

    def clear_error_msg_bottom(self):
        blank_txt = tk.StringVar()
        blank_txt.set(" " * 100)
        blank = tk.Message(self.root_window, textvariable=blank_txt, aspect=10000)
        blank.place(x=25, y=600)

    def clear_msg_top(self):
        blank = tk.StringVar()
        blank.set(" " * 80)
        blank_msg = tk.Message(self.root_window, textvariable=blank, aspect=10000, fg="#af290c")
        blank_msg.place(x=880, y=33)

    def start(self):
        """
        Starts the mainloop of the gui.
        :return: None.
        """
        self.root_window.mainloop()

    def next(self):
        """
        Refreshes the gui and sets it up for the next image.
        :return: None.
        """
        if self.dialog_saved:
            self.root_window.destroy()
            self.__init__()
            self.start()
        else:
            self.clear_error_msg_bottom()
            error_txt = tk.StringVar()
            error_txt.set("You need to save the dialog before proceeding.")
            error_msg = tk.Message(self.root_window, textvariable=error_txt, aspect=10000, fg="#af290c")
            error_msg.place(x=25, y=600)


if __name__ == "__main__":
    gui = GUI()
    gui.start()
