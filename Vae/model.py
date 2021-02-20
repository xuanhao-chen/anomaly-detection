import time,os,sys


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .network import *
from .earlyStopping import EarlyStopping
from .eval_methods import *

##



##
class Generator(nn.Module):

    def __init__(self, opt):
        super(Generator, self).__init__()
        self.encoder = Encoder(opt.ngpu,opt,opt.nz)
        self.decoder = Decoder(opt.ngpu,opt)

    def reparameter(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encoder(x)
        latent_z = self.reparameter(mu, log_var)
        output = self.decoder(latent_z)
        return output, latent_z, mu, log_var


class VAE(VAE_MODEL):


    def __init__(self, opt, train_dataloader, val_dataloader, test_data, label, device):
        super(VAE, self).__init__(opt)

        self.early_stopping = EarlyStopping(opt, patience=opt.patience, verbose=False)

        self.opt = opt
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_data = test_data
        self.label = label
        self.device = device


        self.train_batchsize = opt.train_batchsize
        self.val_batchsize = opt.val_batchsize
        self.test_batchsize = opt.test_batchsize
        self.nz = opt.nz
        self.niter = opt.niter

        self.G = Generator(opt).to(device)
        self.G.apply(weights_init)
        # if not self.opt.istest:
        #     print_network(self.G)


        self.bce_criterion = nn.BCELoss()
        self.mse_criterion=nn.MSELoss()
        self.l1loss = nn.L1Loss()


        self.optimizer_G = optim.Adam(self.G.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))


        self.cur_epoch = 0
        self.input =  None

        self.p_z = None


        #output of generator
        self.mu = None
        self.log_var = None
        self.out_g_fake = None
        self.latent_z = None


        self.loss_g = None
        self.loss_g_rs = None
        self.loss_g_rec = None
        self.loss_g_lat = None




    def train(self):

        self.train_hist = {}
        self.train_hist['per_epoch_time'] = []



        print("Train model.")
        start_time = time.time()



        for epoch in range(self.niter):

            self.cur_epoch += 1

            self.train_epoch()

            val_error = self.validate()

            self.early_stopping(val_error, self.G, self.D_rec, self.D_lat)

            print('epoch', epoch)

            if self.early_stopping.early_stop:
                print('train finished with early stopping')
                break

        if not self.early_stopping.early_stop:
            print('train finished with total epochs')
            self.save_weight_GD()

        total_train_time = time.time() - start_time

        self.save(self.train_hist)

        return total_train_time, np.mean(self.train_hist['per_epoch_time'])



    def train_epoch(self):

        epoch_start_time = time.time()
        self.G.train()


        epoch_iter = 0

        for data in self.train_dataloader:

            self.train_batchsize = data.size(0)
            #self.total_steps += self.train_batchsize
            epoch_iter += 1

            self.input = data.permute([0,2,1]).float().to(self.device)



            self.optimize()

            loss = self.get_errors()

            if (epoch_iter  % self.opt.print_freq) == 0:

                print("Epoch: [%d] [%4d/%4d] G_loss(R/L/ALL): %.6f/%.6f/%.6f" %
                      ((self.cur_epoch), (epoch_iter), self.train_dataloader.dataset.__len__()
                       // self.train_batchsize,
                       loss["loss_g_rs"], loss["loss_g_lat"], loss["loss_g"]))


        self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)



    ##
    def optimize(self):


        self.update_g()


    def update_g(self):

        self.G.zero_grad()

        self.out_g_fake, self.latent_z, self.mu, self.log_var = self.G(self.input)


        self.loss_g_rs = self.l1loss(self.out_g_fake, self.input)

        self.loss_g_lat = -0.5 * torch.sum(1 + self.log_var - torch.exp(self.log_var) - self.mu ** 2)

        self.loss_g = self.loss_g_rs + self.opt.w_lat * self.loss_g_lat
        self.loss_g.backward()
        self.optimizer_G.step()


    ##
    def get_errors(self):

        loss = {
                'loss_g': self.loss_g.item(),
                'loss_g_lat': self.loss_g_lat.item(),
                'loss_g_rs': self.loss_g_rs.item(),
                  }

        return loss

        ##

    def validate(self):
        '''
        validate by validation loss
        :return: l1loss
        '''
        l1loss = nn.L1Loss()
        self.G.eval()

        loss = []
        with torch.no_grad():
            for i, data in enumerate(self.val_dataloader, 0):

                input_data = data.permute([0,2,1]).float().to(self.device)
                fake, _, _, _ = self.G(input_data)
                loss.append(l1loss(input_data, fake).cpu().numpy())

            val_loss = np.mean(loss)

        return val_loss

    def predict(self, dataloader, test_label, scale=True):

        with torch.no_grad():

            collector = []
            pred_time = []

            for i, data in enumerate(dataloader, 0):

                start_time = time.time()

                input_data = data.permute([0,2,1]).float().to(self.device)

                fake, _, _, _ = self.G(input_data)

                fake = fake.type(torch.DoubleTensor)
                data = data.type(torch.DoubleTensor)

                rec_error = torch.sum(torch.abs((fake.permute([0,2,1]) - data)), dim=2)

                collector.append(rec_error[:, -1])

                pred_time.append(time.time() - start_time)

            score = np.concatenate(collector, axis=0)

            # Scale error vector between [0, 1]
            if scale:
                score = (score - np.min(score)) / (np.max(score) - np.min(score))

            y_ = test_label

            y_pred = score

            if y_ is not None and len(y_) > len(y_pred):
                y_ = y_[-len(y_pred):]

            return y_, y_pred, np.mean(pred_time)

    def eval_result(self, dataloader, test_label):

        self.G.eval()

        y_t, y_pred, test_time = self.predict(dataloader, test_label, scale=True)

        t, th = bf_search(y_pred, y_t)

        print('best_f1:', t[0], 'pre:', t[1], 'rec:', t[2], 'TP:', t[3], 'TN:', t[4], 'FP:', t[5], 'FN:', t[6],
              'latency:', t[7], 'threshold:', th)

        return t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], test_time
