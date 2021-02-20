import numpy as np

from sklearn.preprocessing import MinMaxScaler





def proprocess(df):
    df = np.asarray(df, dtype=np.float32)

    if len(df.shape) == 1:
        raise ValueError("Data must be 2-D array")

    if np.any(sum(np.isnan(df)) != 0):
        print("Data contains nan. Will be repalced with 0")

        df = np.nan_to_num()

    df = MinMaxScaler().fit_transform(df)

    print("Data is normalized [0,1]")

    return df


def read_train_data(seq_length, file = '', step=1, valid_portition=0.3):

    values = []

    df = np.load('../datasets/train/' + file, allow_pickle=True)
    print(df.shape)

    (whole_len, whole_dim) = df.shape


    values = proprocess(df)

    flag = 1

    n = int(len(values) * valid_portition)

    if n > seq_length:#the length of validation set must be larger than the length of window size
        train, val = values[:-n], values[-n:]

        if step == 1:
            num_samples_train = (train.shape[0] - seq_length) + 1
            num_samples_val = (val.shape[0] - seq_length) + 1

        else:
            num_samples_train = (train.shape[0] - seq_length) // step
            num_samples_val = (val.shape[0] - seq_length) // step

        temp_train = np.empty([num_samples_train, seq_length, whole_dim])

        temp_val = np.empty([num_samples_val, seq_length, whole_dim])

        for i in range(num_samples_train):
            for j in range(train.shape[1]):
                temp_train[i, :, j] = train[(i*step):(i*step + seq_length), j]

        for i in range(num_samples_val):
            for j in range(val.shape[1]):
                temp_val[i, :, j] = val[(i*step):(i*step + seq_length), j]

        train_data = temp_train

        val_data = temp_val

    else:
        train = values

        if train.shape[0] - seq_length <= 0:
            flag = 0
            train_data = None

            val_data = None

        else:

            if step == 1:
                num_samples_train = (train.shape[0] - seq_length) + 1


            else:
                num_samples_train = (train.shape[0] - seq_length) // step


            temp_train = np.empty([num_samples_train, seq_length, whole_dim])


            for i in range(num_samples_train):
                for j in range(train.shape[1]):
                    temp_train[i, :, j] = train[(i * step):(i * step + seq_length), j]

            train_data = temp_train

            val_data = train_data

    return train_data, val_data, flag



def read_test_data(seq_length, file = ''):

    df = np.load('../datasets/test/' + file, allow_pickle=True)
    label = np.load('../datasets/test_label/' + file, allow_pickle=True).astype(np.float)
    print(df.shape, label.shape)

    (whole_len, whole_dim) = df.shape


    test = proprocess(df)



    num_samples_test = (test.shape[0] - seq_length) + 1

    temp_test = np.empty([num_samples_test, seq_length, whole_dim])


    for i in range(num_samples_test):
        for j in range(test.shape[1]):
                temp_test[i, :, j] = test[(i):(i + seq_length), j]


    test_data = temp_test

    return test_data, label