import pandas as pd
import numpy as np
import torch


def prepare_test_data(sample_length, all_labels):
    overlap = 0
    
    label_csv = pd.read_csv('../../processed_data/dev_split_Depression_AVEC2017.csv')
    user_list = list(label_csv['Participant_ID'])

    test_X = []
    test_y = []
    seq_length = []
    user_names = []

    for user in user_list:
        sorted_feature = np.array(np.load('../../processed_data/feature_fc1/' + str(user) + '.npy'))
        starting_idx = 0
        x_tmp = []
        while starting_idx + sample_length < len(sorted_feature):
            sorted_feature_tmp = sorted_feature[starting_idx:starting_idx + sample_length]
            sorted_feature_tmp = torch.Tensor(np.transpose(sorted_feature_tmp))
            x_tmp.append(sorted_feature_tmp)
            starting_idx = starting_idx + sample_length - overlap

        test_X.append(torch.stack((x_tmp)))
        test_y.append(all_labels[user])
        user_names.append(user)

    return test_X, test_y, user_names
