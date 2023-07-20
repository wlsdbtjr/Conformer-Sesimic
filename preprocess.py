from utils import make_dataset, make_pickle_file


def preprocess(data_list, pickle_filename_list):
    dataset = make_dataset(data_list)
    print("Dataset making process is done")
    for data, filename in zip(dataset, pickle_filename_list):
        print("making {}".format(filename))
        make_pickle_file(data, filename)
        print("{} is saved.".format(filename))


if __name__ == '__main__':
    data_list = [
        ['datasets/chunk1.csv', 'datasets/chunk1.hdf5'],
        ['datasets/chunk2.csv', 'datasets/chunk2.hdf5'],
        ['datasets/chunk3.csv', 'datasets/chunk3.hdf5'],
        ['datasets/chunk4.csv', 'datasets/chunk4.hdf5'],
        ['datasets/chunk5.csv', 'datasets/chunk5.hdf5'],
        ['datasets/chunk6.csv', 'datasets/chunk6.hdf5']]

    pickle_filename_list = ['datasets/train_STEAD_data_magnitude.pkl',
                            'datasets/valid_STEAD_data_magnitude.pkl',
                            'datasets/test_STEAD_data_magnitude.pkl']

    preprocess(data_list=data_list, pickle_filename_list=pickle_filename_list)
