import pandas as pd
from sklearn.preprocessing import MinMaxScaler


speed = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
train_person = ['GP1', 'GP2', 'GP3', 'GP5', 'GP6', 'GP7', 'GP8', 'GP9', 'GP10', 'GP11', 'GP12', 'GP13', 'GP14', 'GP15', 'GP16', 'GP17', 'GP18', 'GP19', 'GP20', 'GP21', 'GP22']

hz = 200

raw_data = {s: [] for s in speed}
preprocessed_data = {s: [] for s in speed}
for s in speed:
    for p in train_person:
        file = pd.read_csv(f"dataset\\archive\\data\\{p}_{s}_marker.csv")
        raw_data[s].append(file)

        print(f"added data\\{p}_{s}_marker.csv")

scaler = MinMaxScaler()
preprocessed_data_path = "dataset\\archive\\preprocessed_data"


# 0과 1 사이로 정규화
for s in speed:
    data = raw_data[s]
    i = 1
    for d in data:
        d_scaled = pd.DataFrame(scaler.fit_transform(d), columns=d.columns).dropna()
        location = f"{preprocessed_data_path}\\{s}_{i}.csv"

        d_scaled.to_csv(location, float_format='%.10f', index=False)


        i+=1

print("전처리 끝")