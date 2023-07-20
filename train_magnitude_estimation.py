import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models import MagnitudeEstimator
from datasets import MagnitudeEstimationDataset
from utils import load_pickle_file, LoRAize
import loralib as lora


def train(model, train_loader, valid_loader, criterion, optimizer, epochs, save_path):
    flag = False
    cnt = 0
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    best_loss = 100

    for epoch in range(epochs):
        if flag is True:
            continue

        train_loss = 0.
        for iter, (waveform, target) in enumerate(train_loader):
            model.train()
            model.to(device)
            waveform = waveform.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(waveform).squeeze(1)
            loss = criterion(output.to(torch.float32), target.to(torch.float32))
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().item()

            if iter % 10 == 0 and iter != 0:
                print('Epoch {}, train loss: {:.6f}'.format(epoch, train_loss / iter))

        train_loss /= len(train_loader.dataset)

        valid_loss = 0.
        for waveform, target in valid_loader:
            model.eval()
            waveform = waveform.to(device)
            target = target.to(device)

            with torch.no_grad():
                output = model(waveform).squeeze(1)
                loss = criterion(output.to(torch.float32), target.to(torch.float32))
                valid_loss += loss.detach().item()

        valid_loss /= len(valid_loader.dataset)

        if valid_loss <= best_loss:
            save_model = save_path + 'bestConformer_epoch{}.pth'.format(epoch)
            torch.save(lora.lora_state_dict(model), save_model)
            best_loss = valid_loss
            print("Saved model {}".format(save_model))
            cnt = 0
        else:
            cnt += 1
            if cnt >= 10:
                flag = True

        print("Epoch {}, train loss: {:.6f}, valid loss: {:.6f}".format(
            epoch, train_loss, valid_loss))


def main():
    model = MagnitudeEstimator()
    model.load_state_dict(torch.load('trained_model/magnitude_estimation/train_STEAD/bestConformer_epoch20.pth'))
    model = LoRAize(model=model, r=1)

    train_data_path = 'datasets/train_domestic_data_magnitude_0.5.pkl'
    valid_data_path = 'datasets/valid_domestic_data_magnitude_0.5.pkl'

    train_data = load_pickle_file(train_data_path)
    valid_data = load_pickle_file(valid_data_path)

    train_dataset = MagnitudeEstimationDataset(train_data)
    valid_dataset = MagnitudeEstimationDataset(valid_data)

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=512, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    epochs = 500

    save_path = 'trained_model/magnitude_estimation/lora/'

    print("Start Training...")
    train(model, train_loader, valid_loader, criterion, optimizer, epochs, save_path)


if __name__ == "__main__":
    main()
