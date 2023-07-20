import torch.nn as nn
import pickle
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import loralib as lora
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


def merge_csv_hdf5(csv_file_path, hdf5_file_path):
    csv_file = pd.read_csv(csv_file_path)
    csv_file = csv_file[['trace_name', 'trace_category', 'source_magnitude', 'p_travel_sec']]

    hdf5_file = h5py.File(hdf5_file_path, 'r')['data']
    hdf5_file = pd.DataFrame(hdf5_file.items())
    hdf5_file.columns = ['trace_name', 'waveform']

    merged_df = pd.merge(csv_file, hdf5_file, on='trace_name')

    for idx in range(len(merged_df['waveform'])):
        waveform = np.array(merged_df['waveform'][idx])
        waveform = np.transpose(waveform)  # (6000, 3) -> (3, 6000)

        for j in range(2):
            waveform[j] = (waveform[j] - np.mean(waveform[j])) / (np.std(waveform[j]) + 1e-9)

        merged_df['waveform'][idx] = waveform

        flag = merged_df['trace_category'][idx] == 'noise'
        label = 0 if flag is True else 1  # 0: noise 1: earthquake
        merged_df['trace_category'][idx] = label

        print("csv file: {}, index {} is preprocessed".format(csv_file_path, idx))
    print("{} is preprocessed".format(csv_file_path))

    return merged_df


def concatenate_df(data_list):
    result = pd.DataFrame()
    for [csv_file_path, hdf5_file_path] in data_list:
        merged_df = merge_csv_hdf5(csv_file_path, hdf5_file_path)
        result = pd.concat([result, merged_df])
        print("{} is merged.".format(csv_file_path))
    result.rename(columns={'trace_category': 'label'}, inplace=True)
    return result


def split_dataset(df):
    train_data, temp = train_test_split(df, test_size=0.15, shuffle=True)
    valid_data, test_data = train_test_split(temp, test_size=0.66, shuffle=True)  # train:valid:test = 85:5:10
    return train_data, valid_data, test_data


def make_dataset(data_list):
    df = concatenate_df(data_list)
    train_data, valid_data, test_data = split_dataset(df)
    train_data.reset_index(drop=True, inplace=True)
    valid_data.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)

    return [train_data, valid_data, test_data]


def make_pickle_file(data, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle_file(pkl):
    with open(pkl, 'rb') as f:
        dataset = pickle.load(f)

    return dataset


def make_spectrogram(wave):
    frequencies, times, amplitudes = signal.spectrogram(wave,
                                                        fs=100,
                                                        window='hamming',
                                                        nperseg=128,
                                                        noverlap=None,
                                                        detrend=False,
                                                        scaling='spectrum')

    spectrogram = 20 * np.log10(amplitudes + 1e-9)

    return spectrogram


def make_histogram(pkl, title):
    fontdict = {'fontname': 'Times New Roman',
                'fontsize': 17}

    with open(pkl, 'rb') as f:
        experiment_result = pickle.load(f)
    experiment_result = [x.item() for x in experiment_result]
    mean = np.mean(experiment_result)
    std = np.std(experiment_result)
    plt.title(title)
    plt.hist(experiment_result, range=(-10, 10), bins=100)
    y_min, y_max = plt.ylim()
    plt.ylim(y_min, y_max)
    plt.text(-10, y_max * 0.8, "{}: {:.3f}\n {}: {:.3f}".format(r'$m$', mean, r'$\sigma$', std), fontdict=fontdict)
    plt.xlabel('Prediction Errors (magnitude)')
    plt.ylabel('Frequency')
    plt.show()


def make_scatter_plot(pkl, title):
    fontdict = {'fontname': 'Times New Roman',
                'fontsize': 17}

    with open(pkl, 'rb') as f:
        experiment_result = pickle.load(f)
    [output, target] = experiment_result
    output = [x.cpu().numpy() for x in output]
    target = [x.cpu().numpy() for x in target]
    r2 = r2_score(y_true=target, y_pred=output)

    plt.title(title)
    plt.scatter(target, output, alpha=0.1)
    y_min, y_max = plt.ylim()
    plt.xlim(-0.2, 5.2)
    plt.ylim(-0.2, 5.2)

    plt.text(0, 4.5, "{}: {:.4f}".format(r'$R^2$', r2), fontdict=fontdict)
    plt.xlabel('True magnitude')
    plt.ylabel('Predicted magnitude')
    plt.rcParams['lines.linestyle'] = '--'
    plt.plot(np.arange(-0.2, 5.2), np.arange(-0.2, 5.2), color='black', alpha=0.7)
    plt.show()


def get_all_parent_layers(net, type):
    layers = []

    for name, l in net.named_modules():
        if isinstance(l, type):
            tokens = name.strip().split(".")
            layer = net
            for t in tokens[:-1]:
                if not t.isnumeric():
                    layer = getattr(layer, t)
                else:
                    layer = layer[int(t)]
            layers.append([layer, tokens[-1]])
    return layers


def LoRAize(model, components=['linear'], r=16):
    if 'linear' in components:
        for parent_layer, last_token in get_all_parent_layers(model, nn.Linear):
            prev_module = getattr(parent_layer, last_token)
            bias = False if prev_module.bias is None else True
            module = lora.Linear(prev_module.in_features, prev_module.out_features, bias=bias, r=r)
            for name, prev_params in prev_module.named_parameters():
                module.register_parameter(name, prev_params)
            setattr(parent_layer, last_token, module)
    if 'embedding' in components:
        for parent_layer, last_token in get_all_parent_layers(model, nn.Embedding):
            prev_module = getattr(parent_layer, last_token)
            module = lora.Embedding(prev_module.num_embeddings, prev_module.embedding_dim, r=r)
            for name, prev_params in prev_module.named_parameters():
                module.register_parameter(name, prev_params)
            setattr(parent_layer, last_token, module)
    lora.mark_only_lora_as_trainable(model)
    return model
