import torch
import torch.nn as nn
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

import matplotlib.pyplot as plt


class Generator(nn.Module):
    def __init__(self,sequence_len,out_features,hidden_dim,dropout=0.5):
        super().__init__()
        self.sequence_len = sequence_len
        self.out_features = out_features
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.fc1 = nn.Sequential(
            nn.Linear(sequence_len,(sequence_len//8) * hidden_dim,bias=False),
            nn.BatchNorm1d(1),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.block1 = self.make_conv1d_block(hidden_dim,4 * hidden_dim,upsample=True)
        self.block2 = self.make_conv1d_block(4*hidden_dim,2*hidden_dim,upsample=True)
        self.block3 = self.make_conv1d_block(2*hidden_dim,hidden_dim,upsample=True)

        self.last = nn.Conv1d(hidden_dim,out_features,kernel_size=5,padding="same")


    def make_conv1d_block(self,in_channel,out_channel,kernel=3,upsample=True):
        block = []

        if upsample:
            block.append(nn.Upsample(scale_factor=2))
        
        block.append(nn.Conv1d(in_channel,out_channel,kernel,padding="same"))
        block.append(nn.BatchNorm1d(out_channel))
        block.append(nn.ReLU())
        block.append(nn.Dropout(self.dropout))

        return nn.Sequential(*block)

    def forward(self,noise):
        out = self.fc1(noise)
        out = torch.reshape(out,(-1,self.hidden_dim,self.sequence_len//8))
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.last(out)
        out = torch.reshape(out,(-1,self.out_features,self.sequence_len))
        
        return out
    

class Discriminator(nn.Module):
    def __init__(self,sequence_len,in_features,hidden_dim):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_features,hidden_dim,kernel_size=5,stride=2,padding=2),
            nn.LeakyReLU(0.2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(hidden_dim,hidden_dim*2,kernel_size=5,stride=2,padding=2),
            nn.LeakyReLU(0.2)
        )

        self.last = nn.Linear(hidden_dim*2*sequence_len//4,1)
        self.sigmoid = nn.Sigmoid()

    
    def forward(self,x):
        _x = self.conv1(x)
        _x = self.conv2(_x)
        _x = torch.flatten(_x,start_dim=1)
        _x = self.last(_x)
        _x = self.sigmoid(_x)
        return _x
  

class WGAN:
    def __init__(self,seq_len,features=3,g_hidden=64,d_hidden=64,max_iters=1000,saveDir=None,ckptPath=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Train on {}".format(self.device))

        self.G = Generator(seq_len,features,g_hidden).to(self.device)
        self.D = Discriminator(seq_len,features,d_hidden).to(self.device)

        self.load_ckpt(ckptPath)

        self.lr = 5e-4
        self.n_critic = 5

        self.g_optimizer = torch.optim.RMSprop(self.G.parameters(),lr=self.lr)
        self.d_optimizer = torch.optim.RMSprop(self.D.parameters(),lr=self.lr)

        self.seq_len = seq_len
        self.features = features

        self.sample_size = 2
        self.max_iters = max_iters
        self.saveDir = saveDir
        self.g_hidden = g_hidden
        self.d_hidden = d_hidden

        self.g_loss_history = []
        self.d_loss_history = []

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
            
            d_loss_real = 0
            d_loss_fake = 0
            W_loss = 0

            self.G.train()

            for d_iter in range(self.n_critic):
                self.D.zero_grad()
                self.G.zero_grad()

                real = torch.autograd.Variable(data.__next__()).float().to(self.device)
                batch_size = real.size(0)

                real_label = torch.autograd.Variable(torch.Tensor(batch_size, 1).fill_(1), requires_grad=False)
                fake_label = torch.autograd.Variable(torch.Tensor(batch_size, 1).fill_(0), requires_grad=False)

                d_loss_real = criterion(self.D(real),real_label)

                z = torch.randn(batch_size,1,self.seq_len).to(self.device)
                fake = self.G(z)  
                d_loss_fake = criterion(self.D(fake),fake_label)

                d_loss = d_loss_fake + d_loss_real
                d_loss.backward()

                self.d_optimizer.step()
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
                self.write2board(g_iter,d_loss,g_loss,W_loss,img)


            self.g_loss_history.append(g_loss)
            self.d_loss_history.append(d_loss)

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


    def save_model(self):
        torch.save({"G_param":self.G.state_dict(),"D_param":self.D.state_dict()},
                f"{self.saveDir}/net_G-{self.g_hidden}_D-{self.d_hidden}_ckpt.pth")
    
    def write2board(self,iter,d_loss,g_loss,w_loss,img):
        self.writer.add_scalar('Loss/D_loss',d_loss,iter)
        self.writer.add_scalar('Loss/G_loss',g_loss,iter)
        self.writer.add_scalar('Loss/W_distance',w_loss,iter)
        self.writer.add_image('Samples',img,iter)
    
    def plot_synth(self):
        self.G.eval()
        z = torch.randn(self.sample_size,1,self.seq_len).to(self.device)
        fake = self.G(z).detach().cpu().numpy()
        
        fig = plt.figure(figsize=(15,3))
        for i in range(self.sample_size):
            ax = fig.add_subplot(1,self.sample_size,i+1)
            ax.plot(fake[i,0,:],label="x-axis")
            ax.plot(fake[i,1,:],label="y-axis")
            ax.plot(fake[i,2,:],label="z-axis")
            ax.legend()
            ax.margins(0)

        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(),dtype= np.uint8)
        img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = np.transpose(img,(2,0,1))
        return img




    