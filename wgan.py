#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 9 20:53:01 2020

@author: eshwarprasadb
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import time as t
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os
from utils.tensorboard_logger import Logger
from torchvision import utils

class Generator(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.main_module = nn.Sequential(
            nn.ConvTranspose2d(in_channels=100, out_channels=1024, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=256, out_channels=channels, kernel_size=4, stride=2, padding=1))
        self.output = nn.Tanh()

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)

class Discriminator(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.main_module = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=1024),
            nn.LeakyReLU(0.2, inplace=True))
        self.output = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0))


    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)

    def feature_extraction(self, x):
        x = self.main_module(x)
        return x.view(-1, 1024*4*4)


class WGAN_CP(object):
    def __init__(self, args):
        print("WGAN_CP init model.")
        self.G = Generator(args.channels)
        self.D = Discriminator(args.channels)
        self.C = args.channels
        self.check_cuda(args.cuda)
        self.learning_rate = 0.00005
        self.batch_size = 64
        self.weight_cliping_limit = 0.01
        self.d_optimizer = torch.optim.RMSprop(self.D.parameters(), lr=self.learning_rate)
        self.g_optimizer = torch.optim.RMSprop(self.G.parameters(), lr=self.learning_rate)
        self.logger = Logger('./logs')
        self.logger.writer.flush()
        self.number_of_input_image = 10
        self.generator_iters = args.generator_iters
        self.critic_iter = 5

    def check_cuda(self, cuda_flag=False):
        if cuda_flag:
            self.cuda_index = 0
            self.cuda = True
            self.D.cuda()
            self.G.cuda()
            print("Cuda enabled flag: {}".format(self.cuda))


    def train(self, train_loader):
        self.t_begin = t.time()
        self.data = self.get_infinite_batches(train_loader)
        one = torch.FloatTensor([1])
        mone = one * -1
        if self.cuda:
            one = one.cuda()
            mone = mone.cuda()

        for grad_iter in range(self.generator_iters):
            for p in self.D.parameters():
                p.requires_grad = True
            for d_iter in range(self.critic_iter):
                self.D.zero_grad()
                for p in self.D.parameters():
                    p.data.clamp_(-self.weight_cliping_limit, self.weight_cliping_limit)

                input_image = self.data.__next__()
                if (input_image.size()[0] != self.batch_size):
                    continue

                z = torch.rand((self.batch_size, 100, 1, 1))

                if self.cuda:
                    input_image, z = Variable(input_image.cuda()), Variable(z.cuda())
                else:
                    input_image, z = Variable(input_image), Variable(z)

                disc_real_loss = self.D(input_image)
                disc_real_loss = disc_real_loss.mean(0).view(1)
                disc_real_loss.backward(one)

                if self.cuda:
                    z = Variable(torch.randn(self.batch_size, 100, 1, 1)).cuda()
                else:
                    z = Variable(torch.randn(self.batch_size, 100, 1, 1))
                fake_input_image = self.G(z)
                disc_loss_fake = self.D(fake_input_image)
                disc_loss_fake = disc_loss_fake.mean(0).view(1)
                disc_loss_fake.backward(mone)

                disc_loss = disc_loss_fake - disc_real_loss
                Wasserstein_D = disc_real_loss - disc_loss_fake
                self.d_optimizer.step()

            for p in self.D.parameters():
                p.requires_grad = False  # to avoid computation

            self.G.zero_grad()
            z = Variable(torch.randn(self.batch_size, 100, 1, 1)).cuda()
            fake_input_image = self.G(z)
            grad_loss = self.D(fake_input_image)
            grad_loss = grad_loss.mean().mean(0).view(1)
            grad_loss.backward(one)
            grad_cost = -grad_loss
            self.g_optimizer.step()

            if (grad_iter) % 1000 == 0:
                self.save_model()

                if not os.path.exists('training_result_input_image/'):
                    os.makedirs('training_result_input_image/')
                z = Variable(torch.randn(800, 100, 1, 1)).cuda(self.cuda_index)
                examples = self.G(z)
                examples = examples.mul(0.5).add(0.5)
                examples = examples.data.cpu()[:64]
                grid = utils.make_grid(examples)
                utils.save_image(grid, 'training_result_input_image/img_generatori_iter_{}.png'.format(str(grad_iter).zfill(3)))

                time = t.time() - self.t_begin
                print("Generator iter: {}".format(grad_iter))
                print("Time {}".format(time))

                info = {
                    'Wasserstein distance': Wasserstein_D.data[0],
                    'Loss D': disc_loss.data[0],
                    'Loss G': grad_cost.data[0],
                    'Loss D Real': disc_real_loss.data[0],
                    'Loss D Fake': disc_loss_fake.data[0]

                }

                for tag, value in info.items():
                    self.logger.scalar_summary(tag, value, grad_iter + 1)

                info = {
                    'real_input_image': self.real_input_image(
                            , self.number_of_image),
                    'generated_input_image': self.generate_img(z, self.number_of_input_image)
                }

                for tag, input_image in info.items():
                    self.logger.image_summary(tag, input_image, grad_iter + 1)


        self.t_end = t.time()
        print('Time of training-{}'.format((self.t_end - self.t_begin)))
        self.save_model()

    def evaluate(self, test_loader, Discriminator_path, Generator_path):
        self.load_model(Discriminator_path, Generator_path)
        z = Variable(torch.randn(self.batch_size, 100, 1, 1)).cuda()
        examples = self.G(z)
        examples = examples.mul(0.5).add(0.5)
        examples = examples.data.cpu()
        grid = utils.make_grid(examples)
        print("Grid of 8x8 input_image saved to 'dgan_model_image.png'.")
        utils.save_image(grid, 'dgan_model_image.png')

    def real_input_image(self, input_image, number_of_input_image):
        if (self.C == 3):
            return self.to_np(input_image.view(-1, self.C, 32, 32)[:self.number_of_input_image])
        else:
            return self.to_np(input_image.view(-1, 32, 32)[:self.number_of_input_image])

    def generate_img(self, z, number_of_input_image):
        examples = self.G(z).data.cpu().numpy()[:number_of_input_image]
        generated_input_image = []
        for sample in examples:
            if self.C == 3:
                generated_input_image.append(sample.reshape(self.C, 32, 32))
            else:
                generated_input_image.append(sample.reshape(32, 32))
        return generated_input_image

    def to_np(self, x):
        return x.data.cpu().numpy()

    def save_model(self):
        torch.save(self.G.state_dict(), './generator.pkl')
        torch.save(self.D.state_dict(), './discriminator.pkl')
        print('Models save to ./generator.pkl & ./discriminator.pkl ')

    def load_model(self, Disc_filename, Gen_filename):
        Discriminator_path = os.path.join(os.getcwd(), Disc_filename)
        Generator_path = os.path.join(os.getcwd(), Gen_filename)
        self.D.load_state_dict(torch.load(Discriminator_path))
        self.G.load_state_dict(torch.load(Generator_path))
        print('Generator model loaded from {}.'.format(Generator_path))
        print('Discriminator model loaded from {}-'.format(Discriminator_path))

    def get_infinite_batches(self, data_loader):
        while True:
            for i, (input_image, _) in enumerate(data_loader):
                yield input_image


    def generate_latent_walk(self, number):
        if not os.path.exists('interpolated_input_image/'):
            os.makedirs('interpolated_input_image/')

        number_int = 10
        zen_interp = torch.FloatTensor(1, 100, 1, 1)
        z1 = torch.randn(1, 100, 1, 1)
        z2 = torch.randn(1, 100, 1, 1)
        if self.cuda:
            zen_interp = zen_interp.cuda()
            z1 = z1.cuda()
            z2 = z2.cuda()

        zen_interp = Variable(zen_interp)
        input_image = []
        alpha = 1.0 / float(number_int + 1)
        print(alpha)
        for i in range(1, number_int + 1):
            zen_interp.data = z1*alpha + z2*(1.0 - alpha)
            alpha += alpha
            fake_im = self.G(zen_interp)
            fake_im = fake_im.mul(0.5).add(0.5) #denormalize
            input_image.append(fake_im.view(self.C,32,32).data.cpu())

        grid = utils.make_grid(input_image, nrow=number_int )
        utils.save_image(grid, 'interpolated_input_image/interpolated_{}.png'.format(str(number).zfill(3)))
        print("Saved interpolated input_image.")