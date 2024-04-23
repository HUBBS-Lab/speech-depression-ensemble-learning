import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score, recall_score

def evaluate_on_train(model, trainLoader):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    model.eval()  # Set the model to evaluation mode

    y_true = []
    y_pred = []
    for i, data in enumerate(trainLoader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        out = F.softmax(model(inputs), dim=1)
        _, predicted = torch.max(out, 1)

        y_pred.extend(predicted.cpu().numpy())
        y_true.extend(torch.argmax(labels, axis=1).cpu().numpy())


    # Calculate additional evaluation metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    recall_macro = recall_score(y_true, y_pred, average='macro')

    # Round metrics to 4 decimal places
    mae = round(mae, 4)
    rmse = round(rmse, 4)
    f1_macro = round(f1_macro, 4)
    recall_macro = round(recall_macro, 4)

    return mae, rmse, f1_macro, recall_macro, y_true, y_pred
