import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import loralib as lora
from models import SeismicEventClassifier
from datasets import ClassificationDataset
from utils import load_pickle_file, LoRAize


def train(model, train_loader, valid_loader, criterion, optimizer, epochs, save_path):
    flag = False
    cnt = 0
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    best_acc = 1e-4

    for epoch in range(epochs):
        if flag is True:
            continue

        train_loss = 0.
        train_acc = 0.

        for iter, (waveform, target) in enumerate(train_loader):
            model.train()
            model.to(device)
            waveform = waveform.to(device)
            target = torch.unsqueeze(target, 0).permute(1, 0).to(device)

            optimizer.zero_grad()
            output = model(waveform)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().item()

            with torch.no_grad():
                prediction = output.argmax(dim=1, keepdim=True)
                _, indices = torch.max(prediction, dim=1)
                train_acc += prediction.eq(target.view_as(prediction)).sum().item()

            if iter % 10 == 0 and iter != 0:
                print('Epoch {}, train loss: {:4f}'.format(epoch, train_loss / iter))

        train_loss /= len(train_loader.dataset)
        train_acc /= len(train_loader.dataset)

        valid_loss = 0.
        valid_acc = 0.
        for waveform, target in valid_loader:
            model.eval()
            waveform = waveform.to(device)
            target = torch.unsqueeze(target, 0).permute(1, 0).to(device)

            with torch.no_grad():
                output = model(waveform)
                valid_loss += criterion(output, target).item()
                prediction = output.argmax(dim=1, keepdim=True)
                valid_acc += prediction.eq(target.view_as(prediction)).sum().item()

        valid_loss /= len(valid_loader.dataset)
        valid_acc /= len(valid_loader.dataset)

        if valid_acc > best_acc:
            save_model = save_path + 'bestConformer_epoch{}.pth'.format(epoch)
            # torch.save(model.state_dict(), save_model)
            torch.save(lora.lora_state_dict(model), save_model)
            best_acc = valid_acc
            cnt = 0
            print("Saved model {}".format(save_model))

        else:
            cnt += 1
            if cnt >= 10:
                flag = True

        print('Epoch {}, train loss: {:.4f}, valid loss: {:.4f}, train acc: {:.4f}, valid acc: {:.4f}'.format(
            epoch, train_loss, valid_loss, train_acc, valid_acc))


def main():
    model = SeismicEventClassifier(num_classes=2)
    # Fine-tuning
    model.load_state_dict(torch.load("trained_model/classification/bestConformer_epoch24.pth"))
    model = LoRAize(model, r=1)

    train_data_path = 'datasets/train_domestic_data_classification_0.5.pkl'
    valid_data_path = 'datasets/valid_domestic_data_classification_0.5.pkl'

    train_data = load_pickle_file(train_data_path)
    valid_data = load_pickle_file(valid_data_path)

    train_dataset = ClassificationDataset(train_data)
    valid_dataset = ClassificationDataset(valid_data)

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=512, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    epochs = 500

    save_path = 'trained_model/classification/lora/'

    print("Start Training...")
    train(model, train_loader, valid_loader, criterion, optimizer, epochs, save_path)


if __name__ == "__main__":
    main()
