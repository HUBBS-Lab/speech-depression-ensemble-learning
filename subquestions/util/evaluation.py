import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score
from collections import Counter

def cnn_evaluate(model, test_X, test_y):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.eval()  # Set the model to evaluation mode

    y_true = []
    y_pred = []
    for i in range(len(test_X)):
        y_true.append(test_y[i])
        x = Variable(test_X[i]).to(device)

        out = F.softmax(model(x), dim=1)
        prob_tmp = out.data.cpu().numpy()
        label_pred = np.argmax(prob_tmp, axis=1)

        counter = Counter(label_pred)
        max_count = max(counter.values())
        modes = [num for num, count in counter.items() if count == max_count]
        average_mode = int(sum(modes) / len(modes))

        y_pred.append(average_mode)
        # y_pred.append(weighted_sum)

    # Calculate additional evaluation metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    f1_macro = f1_score(y_true, y_pred, average='macro')

    # Round metrics to 4 decimal places
    mae = round(mae, 4)
    rmse = round(rmse, 4)
    f1_macro = round(f1_macro, 4)

    return (mae, rmse, f1_macro, y_true, y_pred)