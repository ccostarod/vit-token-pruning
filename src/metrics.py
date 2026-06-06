import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def calculate_metrics(outputs, targets):
    
    preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()

    acc = accuracy_score(targets, preds)

    macro_f1 = f1_score(targets, preds, average='macro', zero_division=0)

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
    }

def get_predictions_and_targets(outputs, targets):
    preds = torch.argmax(outputs, dim=1)

    preds_np = preds.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()

    return preds_np, targets_np

def calculate_confusion_matrix(y_true, y_pred, labels=None, normalize=None):
    return confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
        normalize=normalize,
    )


