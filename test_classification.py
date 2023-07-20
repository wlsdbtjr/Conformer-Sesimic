import torch
import numpy as np
from torch.utils.data import DataLoader
from models import SeismicEventClassifier
from datasets import ClassificationDataset
from utils import load_pickle_file, LoRAize
from sklearn.metrics import confusion_matrix
from tqdm import tqdm


def test(model, test_loader):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.eval()
    model.to(device)
    test_acc = 0.
    concat_mat = np.zeros((2, 2))

    for waveform, target in tqdm(test_loader):
        waveform = waveform.to(device)
        target = target.to(device)
        with torch.no_grad():
            output = model(waveform)
            prediction = output.argmax(dim=1, keepdim=True).squeeze(1).squeeze(1)
            test_acc += prediction.eq(target.view_as(prediction)).sum().item()
            concat_mat += np.asarray(confusion_matrix(target.cpu().numpy(), prediction.cpu().numpy()), dtype=int)

    test_acc /= len(test_loader.dataset)

    print(concat_mat)
    TP, FN, FP, TN = int(concat_mat[1][1]), int(concat_mat[1][0]), int(concat_mat[0][1]), int(concat_mat[0][0])

    accuracy = (TP + TN) / (TP + FN + FP + TN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1score = 2 * precision * recall / (precision + recall)
    print("Accuracy: {:.6f}, Precision: {:.6f}, Recall: {:.6f}, F1 Score: {:.6f}".format(
        accuracy, precision, recall, f1score))


def main():
    model = SeismicEventClassifier(num_classes=2)
    model = LoRAize(model=model, r=1)
    model.load_state_dict(torch.load("trained_model/classification/bestConformer_epoch24.pth"), strict=False)
    model.load_state_dict(torch.load("trained_model/classification/lora/bestConformer_epoch26.pth"), strict=False)

    # test_data_path = 'datasets/test_STEAD_data.pkl'
    test_data_path = 'datasets/test_domestic_data_classification_0.5.pkl'

    test_data = load_pickle_file(test_data_path)
    test_dataset = ClassificationDataset(test_data)

    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True)
    test(model, test_loader)


if __name__ == "__main__":
    main()
