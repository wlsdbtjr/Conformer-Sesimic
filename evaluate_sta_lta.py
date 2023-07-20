from tqdm import tqdm
import numpy as np
from obspy.signal.trigger import classic_sta_lta
from utils import load_pickle_file


def evaluate_sta_lta(test_data, sampling_rate):
    concat_mat = np.zeros((2, 2))

    for idx in tqdm(range(len(test_data))):
        label = test_data['label'].iloc[idx]
        waveform = test_data['waveform'].iloc[idx]

        E, N, Z = waveform

        sta, lta = int(5 * sampling_rate), int(10 * sampling_rate)

        cft_e = classic_sta_lta(E, sta, lta)
        cft_n = classic_sta_lta(N, sta, lta)
        cft_z = classic_sta_lta(Z, sta, lta)
        cft = np.concatenate([cft_e, cft_n, cft_z], axis=0)

        threshold = 2 * np.mean(cft)
        if max(cft) >= threshold:
            flag = 1
        else:
            flag = 0

        if label == 0 and flag == 0:        # TN
            concat_mat[0][0] += 1

        elif label == 0 and flag == 1:      # FP
            concat_mat[0][1] += 1

        elif label == 1 and flag == 0:      # FN
            concat_mat[1][0] += 1

        elif label == 1 and flag == 1:      # TP
            concat_mat[1][1] += 1

    TP, FN, FP, TN = int(concat_mat[1][1]), int(concat_mat[1][0]), int(concat_mat[0][1]), int(concat_mat[0][0])

    accuracy = (TP + TN) / (TP + FN + FP + TN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1score = 2 * precision * recall / (precision + recall)

    return concat_mat, accuracy, precision, recall, f1score


def main():
    test_data_list = ['datasets/classification/test_domestic_data_classification_0.5.pkl']
    sampling_rate = 100

    for pickle_file in test_data_list:
        test_data = load_pickle_file(pickle_file)
        test_data_name = pickle_file.split('/')[-1]
        concat_mat, accuracy, precision, recall, f1score = evaluate_sta_lta(test_data=test_data,
                                                                            sampling_rate=sampling_rate)

        print("Test Dataset: {}".format(test_data_name))
        print(concat_mat)
        print("Accuracy: {:.6f}, Precision: {:.6f}, Recall: {:.6f}, F1 Score: {:.6f}".format(
            accuracy, precision, recall, f1score))


if __name__ == '__main__':
    main()
