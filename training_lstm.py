import pandas
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out)
        return out


class CustomDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pandas.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_duration = self.data.iloc[idx]['input_duration']
        input_difficulty = self.data.iloc[idx]['input_difficulty']
        input_level = self.data.iloc[idx]['input_level']
        output_duration = self.data.iloc[idx]['output_duration']
        name = self.data.iloc[idx]['name']
        input_onset = self.data.iloc[idx]['input_onset_0':'input_onset_49'].values
        output_columns = self.data.iloc[idx]['output_columns_0':'output_columns_399'].values

        sample = {'input_duration': input_duration, 'input_difficulty': input_difficulty, 'input_level': input_level,
                  'output_duration': output_duration, 'name': name, 'input_onset': input_onset,
                  'output_columns': output_columns}

        return sample


if __name__ == "__main__":
    # 모델 초기화
    input_size = 52  # 입력 데이터의 크기 (50개의 onset 데이터 + level + difficulty)
    hidden_size = 128  # LSTM의 히든 레이어 크기
    num_layers = 2  # LSTM의 레이어 수
    output_size = 400  # 출력 데이터의 크기 (8개의 lane * 50개의 시간 단계)
    model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)

    # 손실 함수와 최적화 함수 정의
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # data from training_set.csv
    train_dataset = CustomDataset('training_set_backfront.csv')
    train_loader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)

    # 학습 루프
    num_epochs = 1000
    for epoch in range(num_epochs):
        for i, (input_seq, target_seq) in enumerate(train_loader):
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)

            # 순전파
            optimizer.zero_grad()
            output_seq = model(input_seq)

            # 손실 계산 및 역전파
            loss = criterion(output_seq, target_seq)
            loss.backward()
            optimizer.step()

            # 로그 출력
            if (i + 1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, len(train_loader), loss.item()))
