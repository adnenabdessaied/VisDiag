#!/usr/bin/env python
__author__ = "Mohamed Adnen Abdessaied"
__version__ = "1.0"
__maintainer__ = "Mohamed Adnen Abdessaied"
__email__ = "adnenabdessayed@gmail.com"
__status__ = "Implementation"


def compute_accuracy(output, gt, mode):
    """
    This function computes the accuracies per batch when the discriminative decoder is used.
    :param output: Output of the decoder per batch.
    :param gt: Ground truth indices per batch
    :param mode: The mode of the experiment: train or validation
    :return: a dict of accuracies using different measures. in_top_i means that the gt answer is in top i predicted
             answers.
    """
    size_ = output.size()

    # Detach output tensor to avoid memory leaks
    output = output.detach()

    _, idx = output.sort(dim=1, descending=True)

    in_top_1 = [gt[i] in idx[i][:1] for i in range(size_[0])]
    in_top_3 = [gt[i] in idx[i][:3] for i in range(size_[0])]
    in_top_5 = [gt[i] in idx[i][:5] for i in range(size_[0])]

    accuracies = {"in_top_1_" + mode: sum(in_top_1) / (size_[0]),
                  "in_top_3_" + mode: sum(in_top_3) / (size_[0]),
                  "in_top_5_" + mode: sum(in_top_5) / (size_[0])}
    return accuracies


