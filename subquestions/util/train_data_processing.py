import pandas as pd
import numpy as np
import torch


def prepare_training_data(all_labels, augmented_values, sample_length):
    label_csv = pd.read_csv('../processed_data/train_split_Depression_AVEC2017.csv')
    user_list = list(label_csv['Participant_ID'])

    X = []
    y = []

    for user in user_list:
        aug_ratio = augmented_values[all_labels[user]]

        top_k_reserve = 21
        marked_uttr = 6
        feature = np.array(np.load('../processed_data/feature_fc1/' + str(user) + '.npy'))
        saliency = list(np.load('../processed_data/saliency/' + str(user) + '.npy'))
        select_range = range(len(saliency) - 42) if len(saliency) != 42 else [0]
        for i in range(aug_ratio):
            start_idx = np.random.choice(select_range, replace=False, size=1)[0]
            feature_part = feature[start_idx:start_idx + 42, :]
            top_idx = np.argsort(saliency[start_idx:start_idx + 42])[:top_k_reserve]
            remaining_idx = list(set(range(42)) - set(top_idx))
            selected_idx = np.random.choice(remaining_idx, replace=False, size=marked_uttr)
            if np.random.rand() < 0.5:
                for idx in selected_idx:
                    feature_part[idx, :] = 0.001
            X.append(torch.Tensor(np.transpose(feature_part)))
            y.append(all_labels[user])

    tensor_X_train = torch.stack((X))
    tensor_y_train = torch.nn.functional.one_hot(torch.tensor(y, dtype=torch.int64), num_classes=4)

    return tensor_X_train, tensor_y_train
