from sklearn.metrics import mean_absolute_error, mean_squared_error, confusion_matrix, classification_report, f1_score
import pandas as pd
import numpy as np

def analyze_output(data):

    if isinstance(data, str):
        # Read the CSV file into a DataFrame
        df = pd.read_csv(data)
        data = df.values
    elif isinstance(data, pd.DataFrame):
        data = data.values
    else:
        raise ValueError("Input data type should be a string (file path) or DataFrame.")
    out = np.transpose(data)
    # Calculate MAE and RMSE between consecutive pairs of lists
    mae_list = []
    rmse_list = []
    for i in range(1, 16, 2):
        mae = mean_absolute_error(out[i], out[i+1])
        rmse = mean_squared_error(out[i], out[i+1], squared=False)
        mae_list.append(mae)
        rmse_list.append(rmse)

    # Calculate bitwise sums
    true_sum = np.sum(out[1:16:2, :], 0)
    pred_sum = np.sum(out[2:17:2, :], 0)

    # print(true_sum)
    print('pred phq-8 score', pred_sum)

    five_way_true = [0 if x <= 4 else 1 if x <= 9 else 2 if x <= 14 else 3 if x <= 19 else 4 for x in true_sum]
    five_way_pred = [0 if x <= 4 else 1 if x <= 9 else 2 if x <= 14 else 3 if x <= 19 else 4 for x in pred_sum]

    five_way_f1 = f1_score(five_way_true, five_way_pred, average='macro', zero_division=0.0)

    # Calculate MAE and RMSE between bitwise sums
    bitwise_mae = mean_absolute_error(true_sum, pred_sum)
    bitwise_rmse = mean_squared_error(true_sum, pred_sum, squared=False)

    # Threshold the bitwise sums
    true_binary = [0 if x < 10 else 1 for x in true_sum]
    pred_binary = [0 if x < 10 else 1 for x in pred_sum]

    report = classification_report(true_binary, pred_binary, zero_division=0, digits=4, output_dict=True)
    report_2 = classification_report(true_binary, pred_binary, zero_division=0, digits=4)

    return (mae_list, rmse_list, bitwise_mae, bitwise_rmse, report, report_2, five_way_f1)

