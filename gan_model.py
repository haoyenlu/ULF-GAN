import torch
import torch.nn as nn



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

class EEG_Generator(nn.Module):
    def __init__(self,sequence_len,out_features,hidden_dim=50):
        super().__init__()
        self.sequence_len = sequence_len
        self.out_features = out_features
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Sequential(
            nn.Linear(sequence_len,(sequence_len//64) * hidden_dim,bias=False),
            nn.LeakyReLU(0.2)
        )

        self.block1 = nn.Sequential(
            self.make_conv1d_block(hidden_dim,hidden_dim,upsample=True),
            self.make_conv1d_block(hidden_dim,hidden_dim,upsample=False)
        )
        self.block2 = nn.Sequential(
            self.make_conv1d_block(hidden_dim,hidden_dim,upsample=True),
            self.make_conv1d_block(hidden_dim,hidden_dim,upsample=False)
        )
        self.block3 = nn.Sequential(
            self.make_conv1d_block(hidden_dim,hidden_dim,upsample=True),
            self.make_conv1d_block(hidden_dim,hidden_dim,upsample=False)
        )
        self.block4 = nn.Sequential(
            self.make_conv1d_block(hidden_dim,hidden_dim,upsample=True),
            self.make_conv1d_block(hidden_dim,hidden_dim,upsample=False)
        )
        self.block5 = nn.Sequential(
            self.make_conv1d_block(hidden_dim,hidden_dim,upsample=True),
            self.make_conv1d_block(hidden_dim,hidden_dim,upsample=False)
        )
        self.block6 = nn.Sequential(
            self.make_conv1d_block(hidden_dim,hidden_dim,upsample=True),
            self.make_conv1d_block(hidden_dim,hidden_dim,upsample=False)
        )


        self.last = nn.Conv1d(hidden_dim,out_features,kernel_size=5,padding="same")


    def make_conv1d_block(self,in_channel,out_channel,kernel=3,upsample=True):
        block = []

        if upsample:
            block.append(nn.Upsample(scale_factor=2))
        
        block.append(nn.Conv1d(in_channel,out_channel,kernel,padding="same"))
        block.append(nn.BatchNorm1d(out_channel))
        block.append(nn.LeakyReLU(0.2))

        return nn.Sequential(*block)

    def forward(self,noise):
        out = self.fc1(noise)
        out = torch.reshape(out,(-1,self.hidden_dim,self.sequence_len//64))
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.last(out)
        out = torch.reshape(out,(-1,self.out_features,self.sequence_len))
        
        return out
    
class EEG_Discriminator(nn.Module):
    def __init__(self,sequence_len,in_features,hidden_dim):
        super().__init__()
        self.first = nn.Conv1d(in_features,hidden_dim,3,padding="same")

        self.block1 = nn.Sequential(
            self.make_conv1d_block(hidden_dim,hidden_dim,downsample=False),
            self.make_conv1d_block(hidden_dim,hidden_dim,downsample=True)
        )
        self.block2 = nn.Sequential(
            self.make_conv1d_block(hidden_dim,hidden_dim,downsample=False),
            self.make_conv1d_block(hidden_dim,hidden_dim,downsample=True)
        )
        self.block3 = nn.Sequential(
            self.make_conv1d_block(hidden_dim,hidden_dim,downsample=False),
            self.make_conv1d_block(hidden_dim,hidden_dim,downsample=True)
        )
        self.block4 = nn.Sequential(
            self.make_conv1d_block(hidden_dim,hidden_dim,downsample=False),
            self.make_conv1d_block(hidden_dim,hidden_dim,downsample=True)
        )
        self.block5 = nn.Sequential(
            self.make_conv1d_block(hidden_dim,hidden_dim,downsample=False),
            self.make_conv1d_block(hidden_dim,hidden_dim,downsample=True)
        )
        self.block6 = nn.Sequential(
            self.make_conv1d_block(hidden_dim,hidden_dim,downsample=False),
            self.make_conv1d_block(hidden_dim,hidden_dim,downsample=True)
        )

        self.last = nn.Linear(hidden_dim*sequence_len//64,1)
        self.sigmoid = nn.Sigmoid()

    def make_conv1d_block(self,in_channel,out_channel,kernel=3,downsample=False):
        block = []

        block.append(nn.Conv1d(in_channel,out_channel,kernel,padding="same"))
        block.append(nn.LeakyReLU(0.2))
        if downsample:
            block.append(nn.AvgPool1d(kernel_size=3,padding=1,stride=2))

        return nn.Sequential(*block)
    
    def forward(self,x):
        _x = self.first(x)
        _x = self.block1(_x)
        _x = self.block2(_x)
        _x = self.block3(_x)
        _x = self.block4(_x)
        _x = self.block5(_x)
        _x = self.block6(_x)
        _x = torch.flatten(_x,start_dim=1)
        _x = self.last(_x)
        _x = self.sigmoid(_x)
        return _x