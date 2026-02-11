from argparse import ArgumentParser

import numpy as np
from matplotlib import pyplot as plt


def get_args():
    parser = ArgumentParser()

    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--train_data", type=str, default="data/gtsrb")
    parser.add_argument("--test_data", type=str, default="data/gtsrb")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--trained", type=str, default="trained")
    parser.add_argument("--logging", type=str, default="tensorboard")
    parser.add_argument("--load_checkpoint", action="store_true")

    return parser.parse_args()

def plot_confusion_matrix(writer, cm, class_names, epoch, mode="recall", fold=None, train=True):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
       mode (string): 'recall' or 'precision'
    """
    cm = cm.astype(float)

    if mode == "recall":
        # TP / (TP + FN)
        denom = cm.sum(axis=1, keepdims=True)
        title = "Confusion Matrix (Recall)"

    elif mode == "precision":
        # TP / (TP + FP)
        denom = cm.sum(axis=0, keepdims=True)
        title = "Confusion Matrix (Precision)"

    else:
        raise ValueError("mode phải là 'recall' hoặc 'precision'")

    denom[denom == 0] = 1

    cm_norm = cm / denom
    cm_norm = np.round(cm_norm, 2)

    figure = plt.figure(figsize=(20, 20))
    plt.imshow(cm_norm, interpolation="nearest", cmap="ocean")
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    threshold = cm_norm.max() / 2.0 if cm_norm.max() > 0 else 0.5

    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            color = "white" if cm_norm[i, j] > threshold else "black"
            plt.text(j, i, cm_norm[i, j],
                     horizontalalignment="center",
                     color=color)

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    s1 = f"Fold {fold}/" if fold is not None else ""
    s2 = "Validation" if train else "Test"

    tag = f"{s1}Confusion Matrix {s2}/{mode}"
    writer.add_figure(tag, figure, epoch)