import torch
import torch.nn as nn
from torch.autograd import Variable
import time as t
import os
from utils.tensorboard_logger import Logger
from utils.inception_score import get_inception_score
from itertools import chain
from torchvision import utils

class Invent(torch.nn.Module):
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

class Evaluater(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        

        self.main_module = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True))
        self.output = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)

    def feature_extraction(self, x):
        x = self.main_module(x)
        return x.view(-1, 1024*4*4)
    
class DCGAN_MODEL(object):
    def __init__(self, args):
        print("DCGAN model initalization.")
        self.G = Invent(args.channels)
        self.D = Evaluater(args.channels)
        self.C = args.channels
        self.loss = nn.BCELoss()
        self.cuda = "False"
        self.cuda_index = 0       
        self.check_cuda(args.cuda)
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.logger = Logger('./logs')
        self.number_of_images = 10

    def check_cuda(self, cuda_flag=False):
        if cuda_flag:
            self.cuda = True
            self.D.cuda(self.cuda_index)
            self.G.cuda(self.cuda_index)
            self.loss = nn.BCELoss().cuda(self.cuda_index)
            print("Cuda enabled flag: ")
            print(self.cuda)

    def train(self, train_loader):
        self.t_begin = t.time()
        Invent_iter = 0
        for epoch in range(self.epochs):
            self.epoch_start_time = t.time()

            for i, (images, _) in enumerate(train_loader):
                # Check if round number of batches
                if i == train_loader.dataset.__len__() // self.batch_size:
                    break
                z = torch.rand((self.batch_size, 100, 1, 1))
                actual_name = torch.ones(self.batch_size)
                duplicate_name = torch.zeros(self.batch_size)

                if self.cuda:
                    images, z = Variable(images).cuda(self.cuda_index), Variable(z).cuda(self.cuda_index)
                    actual_name, duplicate_name = Variable(actual_name).cuda(self.cuda_index), Variable(duplicate_name).cuda(self.cuda_index)
                else:
                    images, z = Variable(images), Variable(z)
                    actual_name, duplicate_name = Variable(actual_name), Variable(duplicate_name)
                outputs = self.D(images)
                d_loss_real = self.loss(outputs, actual_name)
                real_score = outputs
                if self.cuda:
                    z = Variable(torch.randn(self.batch_size, 100, 1, 1)).cuda(self.cuda_index)
                else:
                    z = Variable(torch.randn(self.batch_size, 100, 1, 1))
                dup_images = self.G(z)
                outputs = self.D(dup_images)
                d_loss_fake = self.loss(outputs, duplicate_name)
                fake_score = outputs
                d_loss = d_loss_real + d_loss_fake
                self.D.zero_grad()
                d_loss.backward()
                self.d_optimizer.step()

                if self.cuda:
                    z = Variable(torch.randn(self.batch_size, 100, 1, 1)).cuda(self.cuda_index)
                else:
                    z = Variable(torch.randn(self.batch_size, 100, 1, 1))
                dup_images = self.G(z)
                outputs = self.D(dup_images)
                grad_loss = self.loss(outputs, actual_name)                
                self.D.zero_grad()
                self.G.zero_grad()
                grad_loss.backward()
                self.g_optimizer.step()
                Invent_iter += 1

                if Invent_iter % 1000 == 0:                    
                    print('Epoch-{}'.format(epoch + 1))
                    self.save_model()
                    if not os.path.exists('training_result_images/'):
                        os.makedirs('training_result_images/')                   
                    z = Variable(torch.randn(800, 100, 1, 1)).cuda(self.cuda_index)
                    examples = self.G(z)
                    examples = examples.mul(0.5).add(0.5)
                    examples = examples.data.cpu()[:64]
                    grid = utils.make_grid(examples)
                    utils.save_image(grid, 'training_result_images/img_Inventi_iter_{}.png'.format(str(Invent_iter).zfill(3)))
                    time = t.time() - self.t_begin
                    print("Invent iter: {}".format(Invent_iter))
                    print("Time {}".format(time))

                if ((i + 1) % 100) == 0:
                    print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, grad_loss: %.8f" %
                          ((epoch + 1), (i + 1), train_loader.dataset.__len__() // self.batch_size, d_loss.data[0], grad_loss.data[0]))
                    z = Variable(torch.randn(self.batch_size, 100, 1, 1).cuda(self.cuda_index))

                    info = {
                        'd_loss': d_loss.data[0],
                        'grad_loss': grad_loss.data[0]
                    }
                    
                    for tag, value in info.items():
                        self.logger.scalar_summary(tag, value, Invent_iter)

                    for tag, value in self.D.named_parameters():
                        tag = tag.replace('.', '/')
                        self.logger.histo_summary(tag, self.to_np(value), Invent_iter)
                        self.logger.histo_summary(tag + '/grad', self.to_np(value.grad), Invent_iter)

                    info = {
                        'real_images': self.real_images(images, self.number_of_images),
                        'generated_images': self.generate_img(z, self.number_of_images)
                    }

                    for tag, images in info.items():
                        self.logger.image_summary(tag, images, Invent_iter)
        self.t_end = t.time()
        print('Time of training-{}'.format((self.t_end - self.t_begin)))
        self.save_model()

    def evaluate(self, test_loader, descriminator_path, generator_path):
        self.load_model(descriminator_path, generator_path)
        z = Variable(torch.randn(self.batch_size, 100, 1, 1)).cuda(self.cuda_index)
        examples = self.G(z)
        examples = examples.mul(0.5).add(0.5)
        examples = examples.data.cpu()
        grid = utils.make_grid(examples)
        print("Grid of 8x8 images saved to 'dgan_model_image.png'.")
        utils.save_image(grid, 'dgan_model_image.png')

    def real_images(self, images, number_of_images):
        if (self.C == 3):
            return self.to_np(images.view(-1, self.C, 32, 32)[:self.number_of_images])
        else:
            return self.to_np(images.view(-1, 32, 32)[:self.number_of_images])

    def generate_img(self, z, number_of_images):
        examples = self.G(z).data.cpu().numpy()[:number_of_images]
        generated_images = []
        for sample in examples:
            if self.C == 3:
                generated_images.append(sample.reshape(self.C, 32, 32))
            else:
                generated_images.append(sample.reshape(32, 32))
        return generated_images

    def to_np(self, x):
        return x.data.cpu().numpy()

    def save_model(self):
        torch.save(self.G.state_dict(), './Invent.pkl')
        torch.save(self.D.state_dict(), './Evaluater.pkl')
        print('Models save to ./Invent.pkl & ./Evaluater.pkl ')

    def load_model(self, D_model_filename, G_model_filename):
        descriminator_path = os.path.join(os.getcwd(), D_model_filename)
        generator_path = os.path.join(os.getcwd(), G_model_filename)
        self.D.load_state_dict(torch.load(descriminator_path))
        self.G.load_state_dict(torch.load(generator_path))
        print('Invent model loaded from {}.'.format(generator_path))
        print('Evaluater model loaded from {}-'.format(descriminator_path))

    def generate_latent_walk(self, number):
        if not os.path.exists('interpolated_images/'):
            os.makedirs('interpolated_images/')
        number_int = 10
        zen_interp = torch.FloatTensor(1, 100, 1, 1)
        z1 = torch.randn(1, 100, 1, 1)
        z2 = torch.randn(1, 100, 1, 1)
        if self.cuda:
            zen_interp = zen_interp.cuda()
            z1 = z1.cuda()
            z2 = z2.cuda()
        zen_interp = Variable(zen_interp)
        images = []
        alpha = 1.0 / float(number_int + 1)
        print(alpha)
        for i in range(1, number_int + 1):
            zen_interp.data = z1*alpha + z2*(1.0 - alpha)
            alpha += alpha
            Image_fake = self.G(zen_interp)
            Image_fake = Image_fake.mul(0.5).add(0.5) #denormalize
            images.append(Image_fake.view(self.C,32,32).data.cpu())
        grid = utils.make_grid(images, nrow=number_int )
        utils.save_image(grid, 'interpolated_images/interpolated_{}.png'.format(str(number).zfill(3)))
        print("Saved Images/interpolated_{}.".format(str(number).zfill(3)))