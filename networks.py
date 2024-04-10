import torch
import torch.nn as nn
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

import matplotlib.pyplot as plt

from models import Generator, Discriminator, EEG_Generator , EEG_Discriminator

class GAN:
    def __init__(self,seq_len,features=3,n_critic=3,lr=5e-4,g_hidden=64,d_hidden=64,max_iters=1000,
                 saveDir=None,ckptPath=None,prefix="T01",use_spectral=False,use_eeg=False):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Train on {}".format(self.device))

        self.use_eeg = use_eeg
    
        if self.use_eeg:
            self.G = EEG_Generator(seq_len,features,g_hidden).to(self.device)
            self.D = EEG_Discriminator(seq_len,features,d_hidden).to(self.device)
        else:
            self.G = Generator(seq_len,features,g_hidden).to(self.device)
            self.D = Discriminator(seq_len,features,d_hidden).to(self.device)

        self.load_ckpt(ckptPath)

        self.lr = lr
        self.n_critic = n_critic

        self.g_optimizer = torch.optim.RMSprop(self.G.parameters(),lr=self.lr)
        self.d_optimizer = torch.optim.RMSprop(self.D.parameters(),lr=self.lr)

        self.seq_len = seq_len
        self.features = features

        self.sample_size = 2
        self.max_iters = max_iters
        self.saveDir = saveDir
        self.g_hidden = g_hidden
        self.d_hidden = d_hidden
        self.prefix = prefix
        self.use_spectral = use_spectral

        self.writer = SummaryWriter()

    def train(self,dataloader,show_summary=False):
        if show_summary:
            summary(self.G,(1,self.seq_len))
            summary(self.D,(self.features,self.seq_len))
        

        data = self.get_infinite_batch(dataloader)
        batch_size = 4

        
        criterion = nn.BCELoss()


        for g_iter in range(self.max_iters):
            for p in self.D.parameters():
                p.requires_grad = True
            

            self.G.train()

            for d_iter in range(self.n_critic):
                self.D.zero_grad()
                self.G.zero_grad()

                real = torch.autograd.Variable(data.__next__()).float().to(self.device)
                batch_size = real.size(0)

                real_label = torch.autograd.Variable(torch.Tensor(batch_size, 1).fill_(1), requires_grad=False).to(self.device)
                fake_label = torch.autograd.Variable(torch.Tensor(batch_size, 1).fill_(0), requires_grad=False).to(self.device)

                d_loss_real = criterion(self.D(real),real_label)

                z = torch.randn(batch_size,1,self.seq_len).to(self.device)
                fake = self.G(z)  
                d_loss_fake = criterion(self.D(fake),fake_label)

                d_loss = d_loss_fake + d_loss_real

                if self.use_spectral: 
                    sp_loss = self.spectral_loss(real,fake)
                    d_loss += 0.5 * sp_loss

                d_loss.backward()

                self.d_optimizer.step()
                if self.use_spectral:
                    print(f'Discriminator iteration: {d_iter}/{self.n_critic}, loss_fake: {d_loss_fake}, loss_real: {d_loss_real}, spectral_loss: {sp_loss}')
                else:
                    print(f'Discriminator iteration: {d_iter}/{self.n_critic}, loss_fake: {d_loss_fake}, loss_real: {d_loss_real}')

            self.G.zero_grad()
            self.D.zero_grad()

            z = torch.randn(batch_size,1,self.seq_len).to(self.device)
            fake = self.G(z)
            g_loss = criterion(self.D(fake),real_label)
            g_loss.backward()

            self.g_optimizer.step()
            print(f'Generator iteration: {g_iter}/{self.max_iters}, g_loss: {g_loss}')

            if g_iter % 50 == 0:
                self.save_model()
                img = self.plot_synth()
                self.write2board(g_iter,d_loss,g_loss,img)


            torch.cuda.empty_cache()

        self.save_model()
        print("Finished Training!!")
    

    def load_ckpt(self,ckptPath):
        if ckptPath:
            print("Load Checkpoint....")
            ckpt = torch.load(ckptPath,map_location=self.device)
            self.G.load_state_dict(ckpt['G_param'])
            self.D.load_state_dict(ckpt['D_param'])

    def get_infinite_batch(self,dataloader):
        while True:
            for data in dataloader:
                yield data

    def generate_samples(self,sample_size):
        z = torch.randn(sample_size,1,self.seq_len).to(self.device)
        fakes = self.G(z).detach().cpu().numpy()
        
        return fakes
    
    def spectral_loss(self,real,fake):
        fake_fft = torch.abs(torch.fft.rfft(fake))
        real_fft = torch.abs(torch.fft.rfft(real))
        loss = torch.mean(torch.square(real_fft - fake_fft))
        return loss

    def save_model(self):
        torch.save({"G_param":self.G.state_dict(),"D_param":self.D.state_dict()},
                f"{self.saveDir}/{self.prefix}_net_G{self.g_hidden}_D{self.d_hidden}_ckpt.pth")
    
    def write2board(self,iter,d_loss,g_loss,img):
        self.writer.add_scalar('Loss/D_loss',d_loss,iter)
        self.writer.add_scalar('Loss/G_loss',g_loss,iter)
        self.writer.add_image('Samples',img,iter)
    
    def plot_synth(self):
        self.G.eval()
        z = torch.randn(self.sample_size,1,self.seq_len).to(self.device)
        fake = self.G(z).detach().cpu().numpy()
        
        fig = plt.figure(figsize=(15,3))
        for i in range(self.sample_size):
            ax = fig.add_subplot(1,self.sample_size,i+1)
            for j in range(self.features):
                ax.plot(fake[i,j,:])
            ax.margins(0)

        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(),dtype= np.uint8)
        img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = np.transpose(img,(2,0,1))
        return img




    