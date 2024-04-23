import torch.nn as nn
import torch.optim as optim
import torch
from util.model import CNN
from util.train_evaluator import evaluate_on_train
from util.evaluation import cnn_evaluate

def train_model(trainLoader, train_epochs, test_X, test_y):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_path = 'pretrained_model.pth'

    cnn = CNN()
    cnn.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    cnn.to(device)

    cnn.fc2 = nn.Linear(64, 4).to(device)

    for name, param in cnn.named_parameters():
        if name not in ['fc2.bias', 'fc2.weight']:
            param.requires_grad = False


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=0.001)

    for epoch in range(train_epochs):
        cnn.train()
        running_loss = 0.0
        for i, data in enumerate(trainLoader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = cnn(inputs)
            loss = criterion(outputs, torch.argmax(labels, axis=1))  # Assuming labels are in one-hot format
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(trainLoader)
        # print(f"Epoch [{epoch + 1}/{train_epochs}], Loss: {epoch_loss:.4f}")
        mae, rmse, f1_macro, y_true, y_pred = evaluate_on_train(cnn, trainLoader)
        
    mae, rmse, f1_macro, y_true, y_pred = cnn_evaluate(cnn, test_X, test_y)
    return cnn, y_pred
