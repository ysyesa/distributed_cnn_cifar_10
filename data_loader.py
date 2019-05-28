import numpy as np
import pickle


class DataLoader:
    def __init__(self):
        pass

    @staticmethod
    def load_train_data(data_path, num_data=None):
        x, y = None, None
        for i in range(5):
            path = data_path + "data_batch_" + str(i + 1)
            print "PATH: " + path
            f = open(data_path + "data_batch_" + str(i + 1), "rb")
            datadict = pickle.load(f)
            f.close()

            _x = datadict["data"]
            _y = datadict["labels"]

            _x = np.array(_x, dtype=float) / 255
            _x = _x.reshape([-1, 3, 32, 32])
            _x = _x.transpose([0, 2, 3, 1])
            _x = _x.reshape(-1, 32 * 32 * 3)

            if x is None:
                x, y = _x, _y
            else:
                x = np.concatenate((x, _x), axis=0)
                y = np.concatenate((y, _y), axis=0)

        if num_data is None:
            return x, DataLoader.__convert_to_one_hot_encoding(y)
        else:
            return x[:num_data], DataLoader.__convert_to_one_hot_encoding(y)[:num_data]

    @staticmethod
    def load_test_data(data_path, num_data=None):
        f = open(data_path + "test_batch", "rb")
        datadict = pickle.load(f)
        f.close()

        x = datadict["data"]
        y = np.array(datadict['labels'])

        x = np.array(x, dtype=float) / 255.0
        x = x.reshape([-1, 3, 32, 32])
        x = x.transpose([0, 2, 3, 1])
        x = x.reshape(-1, 32 * 32 * 3)

        if num_data is None:
            return x, DataLoader.__convert_to_one_hot_encoding(y)
        else:
            return x[:num_data], DataLoader.__convert_to_one_hot_encoding(y)[:num_data]

    @staticmethod
    def __convert_to_one_hot_encoding(labels_dense, num_classes=10):
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

        return labels_one_hot
