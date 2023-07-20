import torch
from torch.utils.data import DataLoader
from models import MagnitudeEstimator
from datasets import MagnitudeEstimationDataset
from utils import load_pickle_file, make_pickle_file, LoRAize
from tqdm import tqdm
from comparison_models import DeeperCRNN, MagNet


def test(model, test_loader):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.eval()
    model.to(device)

    error_results = []
    output_lst = []
    target_lst = []
    l1_loss = 0
    less_than_02 = 0
    less_than_01 = 0
    for waveform, target in tqdm(test_loader):
        waveform = waveform.to(device)
        target = target.to(device)
        with torch.no_grad():
            output = model(waveform).squeeze(1)
            error = output - target
            error.tolist()
            output.tolist()
            target.tolist()

            x1 = [x for x in error if abs(x) <= 0.1]
            x2 = [x for x in error if abs(x) <= 0.2]
            x3 = [abs(x) for x in error]

            less_than_01 += len(x1)
            less_than_02 += len(x2)
            l1_loss += sum(x3)
            error_results += error

            output_lst += output
            target_lst += target

    output_target_lst = [output_lst, target_lst]

    mae = l1_loss / len(test_loader.dataset)
    print("Num of total data: {}, Num of error <= 0.2: {}, Num of error <= 0.1: {}, mae: {}".format(
        len(test_loader.dataset), less_than_02, less_than_01, mae))

    # make_pickle_file(error_results, 'results/magnitude_estimation/error_results.pkl')
    # make_pickle_file(output_target_lst, 'results/magnitude_estimation/output_target_lst.pkl')
    # make_pickle_file(error_results, 'results/magnitude_estimation/domestic_error_results_2.pkl')
    # make_pickle_file(output_target_lst, 'results/magnitude_estimation/domestic_output_target_lst_2.pkl')


def main():
    model = MagnitudeEstimator()
    model = LoRAize(model=model, r=1)
    # model = DeeperCRNN()
    # model = MagNet()
    model.load_state_dict(torch.load("trained_model/magnitude_estimation/train_STEAD/bestConformer_epoch20.pth"), strict=False)
    model.load_state_dict(torch.load("trained_model/magnitude_estimation/lora/bestConformer_epoch66.pth"), strict=False)
    # model.load_state_dict(torch.load("trained_model/magnet/train_STEAD/bestMagnet_epoch56.pth"))
    test_data_path = 'datasets/test_domestic_data_magnitude_0.5.pkl'
    # test_data_path = 'datasets/test_STEAD_data_magnitude.pkl'
    test_data = load_pickle_file(test_data_path)
    test_dataset = MagnitudeEstimationDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)

    test(model, test_loader)


if __name__ == "__main__":
    main()
