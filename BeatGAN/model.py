
import time,os,sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .eval_methods import *
from .network import Encoder,Decoder,AD_MODEL,weights_init,print_network

dirname=os.path.dirname
sys.path.insert(0,dirname(dirname(os.path.abspath(__file__))))


##
class Discriminator(nn.Module):

    def __init__(self, opt):
        super(Discriminator, self).__init__()
        model = Encoder(opt.ngpu,opt,1)
        layers = list(model.main.children())

        self.features = nn.Sequential(*layers[:-1])
        self.classifier = nn.Sequential(layers[-1])
        self.classifier.add_module('Sigmoid', nn.Sigmoid())

    def forward(self, x):
        features = self.features(x)
        features = features
        classifier = self.classifier(features)
        classifier = classifier.view(-1, 1).squeeze(1)

        return classifier, features




##
class Generator(nn.Module):

    def __init__(self, opt):
        super(Generator, self).__init__()
        self.encoder1 = Encoder(opt.ngpu,opt,opt.nz)
        self.decoder = Decoder(opt.ngpu,opt)

    def forward(self, x):
        latent_i = self.encoder1(x)
        gen_x = self.decoder(latent_i)
        return gen_x, latent_i


class BeatGAN(AD_MODEL):


    def __init__(self, opt, train_dataloader, val_dataloader, test_data, label, device):
        super(BeatGAN, self).__init__(opt, train_dataloader, device)
        self.train_dataloader = train_dataloader
        self.test_data = test_data
        self.label = label
        self.device = device
        self.device = device
        self.opt=opt

        self.batchsize = opt.batchsize
        self.nz = opt.nz
        self.niter = opt.niter

        self.G = Generator( opt).to(device)
        self.G.apply(weights_init)
        # if not self.opt.istest:
        #     print_network(self.G)

        self.D = Discriminator(opt).to(device)
        self.D.apply(weights_init)
        # if not self.opt.istest:
        #     print_network(self.D)


        self.bce_criterion = nn.BCELoss()
        self.mse_criterion=nn.MSELoss()


        self.optimizerD = optim.Adam(self.D.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizerG = optim.Adam(self.G.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))


        self.total_steps = 0
        self.cur_epoch=0


        self.input = None

        self.gt    = torch.empty(size=(self.batchsize,), dtype=torch.long, device=self.device)
        self.fixed_input = torch.empty(size=(self.batchsize, self.opt.nc, self.opt.window_size), dtype=torch.float32, device=self.device)
        self.real_label = 1
        self.fake_label= 0


        self.out_d_real = None
        self.feat_real = None

        self.fake = None
        self.latent_i = None
        self.out_d_fake = None
        self.feat_fake = None

        self.err_d_real = None
        self.err_d_fake = None
        self.err_d = None

        self.out_g = None
        self.err_g_adv = None
        self.err_g_rec = None
        self.err_g = None


    def train(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []




        print("Train model.")
        start_time = time.time()
        best_auc=0
        best_auc_epoch=0

        for epoch in range(self.niter):

            self.cur_epoch += 1

            self.train_epoch()

            self.save_weight_GD()

        total_train_time = time.time() - start_time

        self.save(self.train_hist)

        # self.save_loss(self.train_hist)

        return total_train_time, np.mean(self.train_hist['per_epoch_time'])


    def train_epoch(self):

        epoch_start_time = time.time()
        self.G.train()
        self.D.train()
        epoch_iter = 0
        for data in self.train_dataloader:

            self.batchsize = data.shape[0]

            self.total_steps += self.batchsize
            epoch_iter += 1

            self.input = data.permute([0, 2, 1]).float().to(self.device)

            self.optimize()

            errors = self.get_errors()

            self.train_hist['D_loss'].append(errors["err_d"])
            self.train_hist['G_loss'].append(errors["err_g"])

            if (epoch_iter  % self.opt.print_freq) == 0:

                print("Epoch: [%d] [%4d/%4d] D_loss(R/F): %.6f/%.6f, G_loss: %.6f" %
                      ((self.cur_epoch), (epoch_iter), self.train_dataloader.dataset.__len__() // self.batchsize,
                       errors["err_d_real"], errors["err_d_fake"], errors["err_g"]))
                # print("err_adv:{}  ,err_rec:{}  ,err_enc:{}".format(errors["err_g_adv"],errors["err_g_rec"],errors["err_g_enc"]))


        self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)


    ##
    def optimize(self):

        self.update_netd()
        self.update_netg()

        # If D loss too low, then re-initialize netD
        if self.err_d.item() < 5e-6:
            self.reinitialize_netd()

    def update_netd(self):
        ##

        self.D.zero_grad()
        # --
        # Train with real

        self.out_d_real, self.feat_real = self.D(self.input)
        # --
        # Train with fake

        self.fake, self.latent_i = self.G(self.input)
        self.out_d_fake, self.feat_fake = self.D(self.fake)
        # --


        self.err_d_real = self.bce_criterion(self.out_d_real, torch.full((self.batchsize,), self.real_label, device=self.device))
        self.err_d_fake = self.bce_criterion(self.out_d_fake, torch.full((self.batchsize,), self.fake_label, device=self.device))


        self.err_d=self.err_d_real+self.err_d_fake
        self.err_d.backward()
        self.optimizerD.step()

    ##
    def reinitialize_netd(self):
        """ Initialize the weights of netD
        """
        self.D.apply(weights_init)
        print('Reloading d net')

    ##
    def update_netg(self):
        self.G.zero_grad()

        self.fake, self.latent_i = self.G(self.input)
        self.out_g, self.feat_fake = self.D(self.fake)
        _, self.feat_real = self.D(self.input)


        # self.err_g_adv = self.bce_criterion(self.out_g, self.label)   # loss for ce
        self.err_g_adv=self.mse_criterion(self.feat_fake,self.feat_real)  # loss for feature matching
        self.err_g_rec = self.mse_criterion(self.fake, self.input)  # constrain x' to look like x


        self.err_g =  self.err_g_rec + self.err_g_adv * self.opt.w_adv
        self.err_g.backward()
        self.optimizerG.step()


    ##
    def get_errors(self):

        errors = {'err_d':self.err_d.item(),
                    'err_g': self.err_g.item(),
                    'err_d_real': self.err_d_real.item(),
                    'err_d_fake': self.err_d_fake.item(),
                    'err_g_adv': self.err_g_adv.item(),
                    'err_g_rec': self.err_g_rec.item(),
                  }


        return errors

        ##

    def get_generated_x(self):
        fake = self.G(self.fixed_input)[0]

        return  self.fixed_input.cpu().data.numpy(),fake.cpu().data.numpy()

    ##

    def predict(self, dataloader, test_label, scale=True):

        with torch.no_grad():

            collector = []
            pred_time = []

            for i, data in enumerate(dataloader, 0):
                start_time = time.time()

                input_data = data.permute([0, 2, 1]).float().to(self.device)

                fake, _ = self.G(input_data)

                fake = fake.type(torch.DoubleTensor)
                data = data.type(torch.DoubleTensor)

                rec_error = torch.sum(torch.abs((fake.permute([0, 2, 1]) - data)), dim=2)

                collector.append(rec_error[:, -1])  # 取每个窗口最后一个点的重构误差

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

