import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score, recall_score, classification_report


def cnn_evaluate(model, test_X, test_y, expert, group_router_dic, user_names):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.eval()  # Set the model to evaluation mode

    y_true = []
    y_pred = []
    for i in range(len(test_X)):
        router_user = np.array(group_router_dic[user_names[i]])
        y_true.append(test_y[i])

        if router_user[expert] == 1:
            x = Variable(test_X[i]).to(device)
            out = F.softmax(model(x), dim=1)
            prob_tmp = out.data.cpu().numpy()
            prob_tmp = np.sum(prob_tmp, axis=0)
            prob = prob_tmp / out.shape[0]
            y_pred.append(np.argmax(prob, axis=0)+expert*5)
        else:
            y_pred.append(0)

        
    return (y_true, y_pred)