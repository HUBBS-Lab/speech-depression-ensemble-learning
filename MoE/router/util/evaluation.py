import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score, recall_score, classification_report
import statistics
from pickle5 import pickle

def cnn_evaluate(model, test_X, test_y, user_names):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    model.eval()  # Set the model to evaluation mode

    y_true = []
    y_pred = []
    save_dict = {}
    for i in range(len(test_X)):
        y_true.append(test_y[i])
        x = Variable(test_X[i]).to(device)

        out = F.softmax(model(x), dim=1)
        prob_tmp = out.data.cpu().numpy()
        prob_tmp = np.argmax(prob_tmp, axis=1)
        mode = max([p[0] for p in statistics._counts(prob_tmp)])
        y_pred.append(mode)
        save_dict[user_names[i]] = mode

    file_path = 'group_router_prob.pkl'
    with open(file_path, 'wb') as f:
        pickle.dump(save_dict, f)
    return (y_true, y_pred)