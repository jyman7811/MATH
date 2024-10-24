import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os

# GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_samples = 1000
timestep = 10
features = 25
seed = 0
lines = num_samples * timestep # 행의 개수
time_per_line = 1 / 200  # 200Hz

# 체크포인트 파일 경로
checkpoint_filepath = 'best_model.pth'

np.random.seed(seed)
torch.manual_seed(seed)

test_speed = 1.5

test_person = 21

class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=26, hidden_size=300, batch_first=True)
        self.fc1 = nn.Linear(300, 30)
        self.fc2 = nn.Linear(30, 5)
        self.fc3 = nn.Linear(5, 24)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.softmax(self.fc3(x))
        return x

# 모델 초기화 및 GPU로 이동
model = LSTMModel().to(device)
criterion = nn.MSELoss()


model.load_state_dict(torch.load(checkpoint_filepath))

test_data = pd.read_csv(f"dataset\\archive\\preprocessed_data\\{test_speed}_{test_person}.csv")
t_times = np.linspace(0, test_data.shape[0] * time_per_line, test_data.shape[0])

test_data['Time'] = t_times


t_starting_pos = test_data.iloc[0].drop('Time')
t_times = test_data['Time']

test_X = pd.DataFrame()
for key in t_starting_pos.keys():
    test_X[key] = np.full(lines, t_starting_pos[key])
test_X['Time'] = t_times
test_X['Speed'] = np.full(lines, test_speed)

# X는 시작 좌표와 속도, 시간, Y는 그에 따른 좌표
test_X = [test_X.iloc[i:i + timestep].values for i in range(0, lines, timestep)]
print("X 묶기 완료")

# test_Y = test_data[t_starting_pos.keys()].diff().dropna()
test_Y = test_data[t_starting_pos.keys()]
test_Y = [test_Y.iloc[i:i + timestep].values for i in range(0, lines, timestep)]

model.eval()
with torch.no_grad():
    test_X_tensor = torch.tensor(np.array(test_X), dtype=torch.float32).to(device)
    test_Y_tensor = torch.tensor(np.array(test_Y), dtype=torch.float32).to(device)
    outputs = model(test_X_tensor)
    loss = criterion(outputs, test_Y_tensor)
    accuracy = (outputs.argmax(dim=1) == test_Y_tensor.argmax(dim=1)).float().mean()

print(f"\nLoss: {loss.item():.4f}")
print(f"Accuracy: {accuracy.item():.4f}")
