"""GANomaly
"""
# pylint: disable=C0301,E1101,W0622,C0103,R0902,R0915

##
from collections import OrderedDict
import os
import time
import numpy as np
from tqdm import tqdm

from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils

from .networks import NetG, NetD, weights_init
from .loss import l2_loss
from .eval_methods import *


class BaseModel():
    """ Base Model for ganomaly
    """

    def __init__(self, opt, train_dataloader, val_dataloader, test_data, label, device):
        ##
        # Seed for deterministic behavior
        self.seed(opt.manualseed)

        # Initalize variables.
        self.opt = opt
        self.train_dataloader = train_dataloader
        self.batchsize = opt.batchsize
        self.device = device

    ##


    ##
    def seed(self, seed_value):
        """ Seed

        Arguments:
            seed_value {int} -- [description]
        """
        # Check if seed is default value
        if seed_value == -1:
            return

        # Otherwise seed all functionality
        import random
        random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        np.random.seed(seed_value)
        torch.backends.cudnn.deterministic = True

    ##
    def get_errors(self):
        """ Get netD and netG errors.
        Returns:
            [OrderedDict]: Dictionary containing errors.
        """

        errors = OrderedDict([
            ('err_d', self.err_d.item()),
            ('err_g', self.err_g.item()),
            ('err_g_adv', self.err_g_adv.item()),
            ('err_g_con', self.err_g_con.item()),
            ('err_g_enc', self.err_g_enc.item())])

        return errors

    ##
    def get_current_images(self):
        """ Returns current images.
        Returns:
            [reals, fakes, fixed]
        """

        reals = self.input.data
        fakes = self.fake.data
        fixed = self.netg(self.fixed_input)[0].data

        return reals, fakes, fixed

    ##
    def save_weights(self, epoch):
        """Save netG and netD weights for the current epoch.
        Args:
            epoch ([int]): Current epoch number.
        """

        weight_dir = os.path.join(self.opt.outf, self.opt.name, 'train', 'weights')
        if not os.path.exists(weight_dir): os.makedirs(weight_dir)

        torch.save({'epoch': epoch + 1, 'state_dict': self.netg.state_dict()},
                   '%s/netG.pth' % (weight_dir))
        torch.save({'epoch': epoch + 1, 'state_dict': self.netd.state_dict()},
                   '%s/netD.pth' % (weight_dir))

    ##
    def train_one_epoch(self):
        """ Train the model for one epoch.
        """

        self.netg.train()
        epoch_iter = 0
        epoch_start_time = time.time()
        for data in self.train_dataloader:

            self.batchsize = data.shape[0]

            self.input = data.permute([0, 2, 1]).float().to(self.device)

            self.total_steps += self.batchsize
            epoch_iter += self.batchsize

            self.gt = torch.empty(size=(self.batchsize,), dtype=torch.long, device=self.device)

            self.real_label = torch.ones(size=(self.batchsize,), dtype=torch.float32, device=self.device)
            self.fake_label = torch.zeros(size=(self.batchsize,), dtype=torch.float32, device=self.device)

            # self.optimize()
            self.optimize_params()

            if self.total_steps % self.opt.print_freq == 0:
                errors = self.get_errors()
        self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)


        print(">> Training model %s. Epoch %d/%d" % (self.name, self.epoch + 1, self.opt.niter))
        # self.visualizer.print_current_errors(self.epoch, errors)

    ##
    def train(self):
        """ Train the model
        """

        ##
        # TRAIN
        self.total_steps = 0
        best_auc = 0
        # Train for niter epochs.
        self.train_hist = {}
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []
        print(">> Training model %s." % self.name)
        start_time = time.time()
        for self.epoch in range(self.opt.iter, self.opt.niter):
            # Train for one epoch
            self.train_one_epoch()
        total_train_time = time.time() - start_time
        print(">> Training model %s.[Done]" % self.name)
        return total_train_time, np.mean(self.train_hist['per_epoch_time'])

    ##
    def test(self, dataloader, test_label, scale=True):
        """ Test GANomaly model.
        Args:
            dataloader ([type]): Dataloader for the test set
        Raises:
            IOError: Model weights not found.
        """
        self.netg.eval()
        self.an_scores = torch.zeros(size=(len(test_label),), dtype=torch.float32,
                                     device=self.device)
        with torch.no_grad():
            # Load the weights of netg and netd.


            self.opt.phase = 'test'

            # Create big error tensor for the test set.

            # self.gt_labels = torch.zeros(size=(len(test_label),), dtype=torch.long,
            #                              device=self.device)
            # self.latent_i = torch.zeros(size=(len(test_label), self.opt.nz), dtype=torch.float32,
            #                             device=self.device)
            # self.latent_o = torch.zeros(size=(len(test_label), self.opt.nz), dtype=torch.float32,
            #                             device=self.device)

            # print("   Testing model %s." % self.name)
            self.times = []
            self.total_steps = 0
            epoch_iter = 0

            for i, data in enumerate(dataloader, 0):
                start_time = time.time()

                input_data = data.permute([0, 2, 1]).float().to(self.device)

                self.batchsize = data.shape[0]


                self.total_steps += self.opt.batchsize
                epoch_iter += self.opt.batchsize
                time_i = time.time()

                self.fake, latent_i, latent_o = self.netg(input_data)




                error = torch.mean(torch.pow((latent_i - latent_o), 2), dim=1)
                time_o = time.time()

                self.an_scores[i * self.batchsize: i * self.batchsize + error.size(0)] = error.reshape(
                    error.size(0))
                # self.gt_labels[i * self.batchsize: i * self.batchsize + error.size(0)] = self.gt.reshape(
                #     error.size(0))
                # self.latent_i[i * self.batchsize: i * self.batchsize + error.size(0), :] = latent_i.reshape(
                #     error.size(0), self.opt.nz)
                # self.latent_o[i * self.batchsize: i * self.batchsize + error.size(0), :] = latent_o.reshape(
                #     error.size(0), self.nz)

                self.times.append(time_o - time_i)

                # Save test images.

            # Measure inference time.
            test_time = np.mean(self.times)


            # Scale error vector between [0, 1]
        self.an_scores = (self.an_scores - torch.min(self.an_scores)) / (
                        torch.max(self.an_scores) - torch.min(self.an_scores))

        self.an_scores = self.an_scores.cpu().numpy()


        t, th = bf_search(self.an_scores, test_label)

        return t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], test_time
            # auc, eer = roc(self.gt_labels, self.an_scores)




##
class Ganomaly(BaseModel):
    """GANomaly Class
    """

    @property
    def name(self):
        return 'Ganomaly'

    def __init__(self, opt, train_dataloader, val_dataloader, test_data, label, device):
        super(Ganomaly, self).__init__(opt, train_dataloader, val_dataloader, test_data, label, device)

        # -- Misc attributes
        self.epoch = 0
        self.times = []
        self.total_steps = 0

        ##
        # Create and initialize networks.
        self.netg = NetG(self.opt).to(self.device)
        self.netd = NetD(self.opt).to(self.device)
        self.netg.apply(weights_init)
        self.netd.apply(weights_init)

        ##
        if self.opt.resume != '':
            print("\nLoading pre-trained networks.")
            self.opt.iter = torch.load(os.path.join(self.opt.resume, 'netG.pth'))['epoch']
            self.netg.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netG.pth'))['state_dict'])
            self.netd.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netD.pth'))['state_dict'])
            print("\tDone.\n")

        self.l_adv = l2_loss
        self.l_con = nn.L1Loss()
        self.l_enc = l2_loss
        self.l_bce = nn.BCELoss()

        ##
        # Initialize input tensors.
        self.input = None
        self.label = torch.empty(size=(self.batchsize,), dtype=torch.float32, device=self.device)


        ##
        # Setup optimizer
        if self.opt.isTrain:
            self.netg.train()
            self.netd.train()
            self.optimizer_d = optim.Adam(self.netd.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            self.optimizer_g = optim.Adam(self.netg.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))

    ##
    def forward_g(self):
        """ Forward propagate through netG
        """
        self.fake, self.latent_i, self.latent_o = self.netg(self.input)

    ##
    def forward_d(self):
        """ Forward propagate through netD
        """
        self.pred_real, self.feat_real = self.netd(self.input)
        self.pred_fake, self.feat_fake = self.netd(self.fake.detach())

    ##
    def backward_g(self):
        """ Backpropagate through netG
        """
        self.err_g_adv = self.l_adv(self.netd(self.input)[1], self.netd(self.fake)[1])
        self.err_g_con = self.l_con(self.fake, self.input)
        self.err_g_enc = self.l_enc(self.latent_o, self.latent_i)
        self.err_g = self.err_g_adv * self.opt.w_adv + \
                     self.err_g_con * self.opt.w_con + \
                     self.err_g_enc * self.opt.w_enc
        self.err_g.backward(retain_graph=True)

    ##
    def backward_d(self):
        """ Backpropagate through netD
        """
        # Real - Fake Loss
        self.err_d_real = self.l_bce(self.pred_real, self.real_label)
        self.err_d_fake = self.l_bce(self.pred_fake, self.fake_label)

        # NetD Loss & Backward-Pass
        self.err_d = (self.err_d_real + self.err_d_fake) * 0.5
        self.err_d.backward()

    ##
    def reinit_d(self):
        """ Re-initialize the weights of netD
        """
        self.netd.apply(weights_init)
        print('   Reloading net d')

    def optimize_params(self):
        """ Forwardpass, Loss Computation and Backwardpass.
        """
        # Forward-pass
        self.forward_g()
        self.forward_d()

        # Backward-pass
        # netg
        self.optimizer_g.zero_grad()
        self.backward_g()
        self.optimizer_g.step()

        # netd
        self.optimizer_d.zero_grad()
        self.backward_d()
        self.optimizer_d.step()
        if self.err_d.item() < 1e-5: self.reinit_d()