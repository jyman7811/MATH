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
np.random.seed(seed)
torch.manual_seed(seed)
lines = num_samples * timestep # 행의 개수

speed = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]

train_person = range(1, 21)

test_speed = 1.0
test_person = 21

raw_data = {s: [] for s in speed}


# 데이터 불러오기
for s in speed:
    for p in train_person:
        file_path = f"dataset\\archive\\preprocessed_data\\{s}_{p}.csv"
        file = pd.read_csv(file_path)
        
        file = file.iloc[:10000]

        raw_data[s].append(file)
        print(f"added {file_path}")

test_data = pd.read_csv(f"dataset\\archive\\preprocessed_data\\{test_speed}_{test_person}.csv")

time_per_line = 1 / 200  # 200Hz

# 데이터 전처리: 시간 데이터 추가
data = {s: [] for s in speed}
for s in speed:
    for raw_d in raw_data[s]:
        # times  =  np.arange(0, d.shape[0] * time_per_line, time_per_line) Unstable
        times = np.linspace(
            0, 
            raw_d.shape[0] * time_per_line,
            raw_d.shape[0]
                    )
        raw_d['Time'] = times
        data[s].append(raw_d)

#t_times = np.arange(0, test_data.shape[0] * time_per_line, time_per_line)
t_times = np.linspace(0, test_data.shape[0] * time_per_line, test_data.shape[0])

test_data['Time'] = t_times

# LSTM 모델 설계
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
optimizer = optim.Adam(model.parameters(), lr=0.00005)

# 체크포인트 파일 경로
checkpoint_filepath = 'best_model.pth'

# 체크포인트 파일이 있으면 가중치 로드
if os.path.exists(checkpoint_filepath):
    print(f"Found checkpoint at {checkpoint_filepath}. Loading weights.")
    model.load_state_dict(torch.load(checkpoint_filepath))
else:
    print(f"No checkpoint found. Starting training from scratch.")

print("Starting Training in 3s...")
time.sleep(1)


# 데이터를 텐서로 변환하고 DataLoader로 준비
for s in speed:
    print(f"NOW SPEED = {str(s)}")
    for raw_d in data[s]:
        starting_pos = raw_d.iloc[0].drop('Time') # 시작 좌표
        times = raw_d['Time']

        X = pd.DataFrame() # X: keys(좀 많음), Time, Speed

        for key in starting_pos.keys():
            X[key] = np.full(lines, starting_pos[key])
        X['Time'] = times
        X['Speed'] = np.full(lines, s)

        # X는 시작 좌표와 속도, 시간, Y는 그에 따른 좌표
        # X = [X.iloc[i:i + timestep].values for i in range(0, lines, timestep)]
        X = [X.iloc[i:i + timestep].values for i in range(0, lines, timestep)]


        Y = raw_d[starting_pos.keys()]
        Y = [Y.iloc[i:i + timestep].values for i in range(0, lines, timestep)]

        # Y = raw_d[starting_pos.keys()].diff().dropna()
        # Y = [Y.iloc[i + timestep].values for i in range(0, num_samples * timestep, timestep)]
        # print("Y 묶기 완료") 이건 변화량 해본거

        X_tensor = torch.tensor(np.array(X), dtype=torch.float32).to(device)
        Y_tensor = torch.tensor(np.array(Y), dtype=torch.float32).to(device)

        # DataLoader를 통해 배치로 학습
        dataset = TensorDataset(X_tensor, Y_tensor)
        loader = DataLoader(dataset, batch_size=128, shuffle=True)

        try:
            model.train()
            for epoch in range(500):
                running_loss = 0.0
                for inputs, targets in loader:
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                if epoch % 100 == 0:
                    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(loader)}")
                # 100번마다 체크포인트 저장
                if epoch % 100 == 0:
                    torch.save(model.state_dict(), checkpoint_filepath)
        except Exception as e:
            print(f"Error during training: {str(e)}")

print("학습 완료")

# 체크포인트로부터 최적의 모델 로드
model.load_state_dict(torch.load(checkpoint_filepath))



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
    test_X_tensor = torch.tensor(test_X, dtype=torch.float32).to(device)
    test_Y_tensor = torch.tensor(test_Y, dtype=torch.float32).to(device)
    outputs = model(test_X_tensor)
    loss = criterion(outputs, test_Y_tensor)
    accuracy = (outputs.argmax(dim=1) == test_Y_tensor.argmax(dim=1)).float().mean()

print(f"\nLoss: {loss.item():.4f}")
print(f"Accuracy: {accuracy.item():.4f}")
