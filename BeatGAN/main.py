import sys
sys.path.append('../')
import os

import torch
from BeatGAN.options import Options
from torch.utils.data import DataLoader
from BeatGAN.data_utils import *
from BeatGAN.model import *
# from dcgan import DCGAN as myModel


device = torch.device("cuda:3" if
torch.cuda.is_available() else "cpu")


opt = Options().parse()
print(opt)
opt.niter = 50
def anomaly_detection(data_type):

    with open('./results/detection_results_beatgan_' + str(data_type) +'.txt', 'w') as f:
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
        opt.dataset = data_type

        if data_type == 'SMAP':
            opt.nc = 25
            opt.step = 1

        elif data_type == 'MSL':
            opt.nc = 55
            opt.step = 1


        elif data_type == 'SMD':
            opt.nc = 38
            opt.step = 1

        elif data_type == 'SWAT':
            opt.nc = 51
            opt.step = 10


        path_train = os.path.join(os.path.dirname(os.getcwd()), "datasets", "train", data_type)


        files = os.listdir(path_train)
        file_number = 0
        for file in files:

            opt.filename = file
            opt.folder = file
            data_name = data_type + '/' + str(file)
            print('file=', data_name)

            samples_train_data, samples_val_data, flag = read_train_data(opt.window_size, file=data_name,
                                                                   step=opt.step)
            if flag == 0:
                continue

            file_number += 1

            print('train samples', samples_train_data.shape)
            train_data = DataLoader(dataset=samples_train_data, batch_size=opt.batchsize, shuffle=True)
            val_data = DataLoader(dataset=samples_val_data, batch_size=opt.batchsize, shuffle=True)

            samples_test_data, test_label = read_test_data(opt.window_size, file=data_name)

            test_data = DataLoader(dataset=samples_test_data, batch_size=opt.batchsize)

            model = BeatGAN(opt, train_data, val_data, test_data, test_label, device)

            train_time, epoch_time = model.train()

            model.load()

            f1, pre, rec, tp, tn, fp, fn, latency, test_time = model.eval_result(test_data, test_label)

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
                  + '\tt_epoch_time=' + str(total_epoch_time))

            f.write(str(data_type) + '\t' + str(file) + '\t' + str(f1) + '\t' + str(pre) + '\t' +
                    str(rec) + '\t' + str(tp) + '\t' + str(tn) + '\t' + str(fp) + '\t' + str(fn) +
                    '\t' + str(train_time) + '\t' + str(epoch_time) + '\t' + str(test_time) +
                    '\t' + str(latency) + '\n')

        f.write('\n')
        f.write('total results' + '\t' + str(data_type) + '\t' + str(total_f1) + '\t' + str(total_pre) + '\t' +
                str(total_rec) + '\t' + str(total_tp) + '\t' + str(total_tn) + '\t' + str(total_fp) + '\t' + str(
            total_fn) +
                '\t' + str(total_train_time) + '\t' + str(total_epoch_time) + '\t' + str(
            total_test_time / file_number / opt.batchsize) +
                '\t' + str(total_latency) + '\n')


    print('finished')

data_type = 'SWAT'

anomaly_detection(data_type)


