import numpy as np
import torch
import torch.nn.functional as F

from t4c22.t4c22_config import class_fractions
from t4c22.metric.masked_crossentropy import get_weights_from_class_fractions

from src.utils.miscs import check_torch_tensor


def label_to_logits(y, num_classes=3, eps=0.0000000000000000001):
    """Coverts label to logits.

    Logits indicate the raw predictions of a classification model, followed by a `softmax` layer to generate probabilities. We therefore use the inverse operation of `softmax` to convert labels to logits.
    >>>  logits = log(y)
    """
    y = check_torch_tensor(y)
    logits = F.one_hot(y.nan_to_num(0).long(), num_classes=num_classes).float()
    logits[logits == 0] = np.log(0 + eps)
    logits[logits == 1] = np.log(1)
    return logits


def proba_to_logits(proba, eps=0.0000000000000000001):
    """Coverts probabilities to logits.

    Logits indicate the raw predictions of a classification model, followed by a `softmax` layer to generate probabilities. We therefore use the inverse operation of `softmax` to convert probabilities to logits.
    >>>  logits = log(proba)
    """
    proba = check_torch_tensor(proba)
    logits = torch.log(proba.nan_to_num(0.0) + eps)
    return logits


def get_city_class_weights(city):
    """Get class weights for city."""
    city_class_fractions = class_fractions[city]
    city_class_weights = torch.tensor(
        get_weights_from_class_fractions(
            [city_class_fractions[c] for c in ["green", "yellow", "red"]]
        )
    ).float()
    return city_class_weights


def calc_loss(city, y, logits):
    """Calculate loss for city."""
    y = check_torch_tensor(y)
    logits = check_torch_tensor(logits)

    city_class_weights = get_city_class_weights(city)
    # compute loss
    loss_f = torch.nn.CrossEntropyLoss(weight=city_class_weights, ignore_index=-1)

    loss = loss_f(logits, y)
    print(f"{city} loss: {loss.cpu().numpy():.3f}")
    return loss


def round_label(y):
    """Round label to nearest class."""
    y = y.nan_to_num(0.0).long()
    return y


def calc_accuracy(y, yhat):
    """Calculate accuracy for city."""
    y = check_torch_tensor(y)
    yhat = check_torch_tensor(yhat)

    valid_idx = y != -1
    num_correct = torch.sum(y[valid_idx] == yhat[valid_idx]).float()
    num_total = torch.sum(valid_idx).float()
    acc = (num_correct / num_total).numpy()

    print(f"Accuracy: {acc * 100:.2f}%")
    return acc


if __name__ == "__main__":
    y = torch.tensor([0, 1, 2, torch.nan, 0, 1, 2]).nan_to_num(-1).long()
    yhat = torch.tensor([0, torch.nan, 2, torch.nan, 1, 2, 1])
    yhat_logits = label_to_logits(yhat)

    print(
        torch.nn.CrossEntropyLoss(reduction="mean")(
            label_to_logits(torch.tensor([1, 2, 0])),
            torch.tensor([1, 2, 0]),
        )
    )

    calc_accuracy(y, yhat)
    calc_loss("london", y, yhat_logits)
