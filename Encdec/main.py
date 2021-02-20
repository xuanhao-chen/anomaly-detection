import sys
sys.path.append('../')
from Encdec.model import *
import numpy as np
import os
from Encdec.eval_methods import *
from sklearn.preprocessing import MinMaxScaler
#https://github.com/datamllab/pyodds/blob/b79ea797dca104b12df5ff03ba701a024e36deac/pyodds/utils/importAlgorithm.py
import time
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


def anomaly_detection(data_type, gpu):

    with open('./results/detection_results_encdec_' + str(data_type) +'.txt', 'w') as f:
        f.write('channel' + '\t' + 'dataset' + '\t' + 'f1' + '\t' + 'pre' + '\t' + 'rec' + '\t' +
                'tp' + '\t' + 'tn' + '\t' + 'fp' + '\t' + 'fn' + '\t' + 'train_time' + '\t' + 'epoch_time' +
                 '\t' +'test_time' + '\t' + 'latency' + '\n')

        total_tp = 0.0
        total_tn = 0.0
        total_fp = 0.0
        total_fn = 0.0
        total_latency = 0.0
        total_train_time = 0.0
        total_test_time = 0.0
        total_epoch_time = 0.0

        path_train = os.path.join(os.path.dirname(os.getcwd()), "datasets", "train", data_type)

        files = os.listdir(path_train)
        file_number = 0
        for file in files:
            file_number += 1

            data_name = data_type + '/' + str(file)
            print('file=', data_name)

            if data_name == 'SMAP/D-12.pkl':
                continue

            contamination = 0.1

            epoch = 20

            model = LSTMED(contamination=contamination, num_epochs=epoch, batch_size=50, lr=1e-3, hidden_size=5,
                           sequence_length=128, train_gaussian_percentage=0.25, gpu=gpu)

            train = np.load('../datasets/train/' + data_name, allow_pickle=True).astype(float)
            print(train.shape)
            train_data = proprocess(train)

            test = np.load('../datasets/test/' + data_name, allow_pickle=True).astype(float)
            print(test.shape)
            test_data = proprocess(test)
            label = np.load('../datasets/test_label/' + data_name, allow_pickle=True).astype(float)
            print(label.shape)


            train_start_time = time.time()
            model.fit(train_data)
            train_end_time = time.time()

            train_time = train_end_time - train_start_time
            epoch_time = train_time / epoch


            test_start_time = time.time()
            outlierness_score = model.decision_function(test_data)
            test_time = (time.time() - test_start_time) / len(outlierness_score)

            score = (outlierness_score - min(outlierness_score)) / (max(outlierness_score) - min(outlierness_score))

            t, th = bf_search(score, label)

            print('best_f1:', t[0], 'pre:', t[1], 'rec:', t[2], 'TP:', t[3], 'TN:', t[4], 'FP:', t[5], 'FN:', t[6],
                  'latency:', t[7], 'threshold:', th)

            f1 = t[0]
            pre = t[1]
            rec = t[2]
            tp = t[3]
            tn = t[4]
            fp = t[5]
            fn = t[6]
            latency = t[7]



            total_tp += tp
            total_tn += tn
            total_fp += fp
            total_fn += fn
            total_latency += latency
            total_pre = total_tp / (total_tp + total_fp)
            total_rec = total_tp / (total_tp + total_fn)
            total_f1 = 2*total_pre*total_rec / (total_pre + total_rec)
            total_train_time += train_time
            total_test_time += test_time
            total_epoch_time += epoch_time

            print(str(data_type) + '\t' + str(file) + '\tf1=' + str(f1) + '\tpre=' + str(pre) +
                  '\trec=' + str(rec) + '\ttp=' + str(tp) + '\ttn=' + str(tn) + '\tfp=' + str(fp) +
                  '\tfn=' + str(fn) + '\tlatency=' + str(latency))

            print('total results:' + '\tt_f1=' + str(total_f1) + '\tt_pre=' + str(total_pre) +
                  '\tt_rec=' + str(total_rec) + '\tt_tp=' + str(total_tp) + '\tt_tn=' + str(total_tn) +
                  '\tt_fp=' + str(total_fp) + '\tt_fn=' + str(total_fn) + '\tt_latency=' + str(total_latency)
                  + '\tt_epoch_time=' + str(total_epoch_time) + '\tt_test_time=' + str(total_test_time / file_number))

            f.write(str(data_type) + '\t' + str(file) + '\t' + str(f1) + '\t' + str(pre) + '\t' +
                    str(rec) + '\t' + str(tp) + '\t' + str(tn) + '\t' + str(fp) + '\t' + str(fn) +
                    '\t' + str(train_time) + '\t' + str(epoch_time) + '\t' + str(test_time) +
                    '\t' + str(latency) + '\n')

        f.write('\n')
        f.write('total results' + '\t' + str(data_type) + '\t' + str(total_f1) + '\t' + str(total_pre) + '\t' +
                str(total_rec) + '\t' + str(total_tp) + '\t' + str(total_tn) + '\t' + str(total_fp) + '\t' + str(
            total_fn) +
                '\t' + str(total_train_time) + '\t' + str(total_epoch_time) + '\t' + str(total_test_time / file_number) +
                '\t' + str(total_latency) + '\n')


    print('finished')



gpu = 0

data_type = 'SWAT'

anomaly_detection(data_type, gpu)






























