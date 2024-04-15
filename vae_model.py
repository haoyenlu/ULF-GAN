import torch
import torch.nn as nn
from torchsummary import summary

class BaseVAE(nn.Module):
    def __init__(self,seq_len,feat_dim,latent_dim):
        super(BaseVAE).__init__()

        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim

        self.encoder = None
        self.decoder = None

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def sampling(self,z_mean,z_log_var):
        return torch.normal(mean=z_mean,std=torch.exp(0.5*z_log_var))
    
    def summary(self):
        summary(self.encoder,(1,self.feat_dim,self.seq_len))
        summary(self.decoder,(1,self.latent_dim))


 
class VariationalAutoencoderConv(BaseVAE): 
    def __init__(self,hidden_layer_size,**kwargs):
        super(VariationalAutoencoderConv).__init__(**kwargs)

        self.hidden_layer_size = hidden_layer_size # a list of out_channel
        self.encoder = self._get_encoder()
        self.decoder = self._get_decoder()
        self.encoder_last_dense_dim = self.calculate_last_dense_dim()

        self.dense_mean = nn.Linear(self.encoder_last_dense_dim,self.latent_dim)
        self.dense_var = nn.Linear(self.encoder_last_dense_dim,self.latent_dim)

        self.first_decoder_dense = nn.Sequential(
            nn.Linear(self.latent_dim,self.encoder_last_dense_dim),
            nn.ReLU()
        )

        self.last_decoder_dense = nn.LazyLinear(self.seq_len * self.feat_dim)

        self.z_mean = None
        self.z_log_var = None


    def _get_encoder(self):
        model = []
        prev_ch = self.feat_dim
        for out_channel in self.hidden_layer_size:
            model.append(nn.Conv1d(prev_ch,out_channel,kernel_size=3,stride=2))
            model.append(nn.ReLU())
            prev_ch = out_channel
        
        model.append(nn.Flatten())

        return nn.Sequential(*model)
    
    def _get_decoder(self):
        model = []
        prev_ch = self.hidden_layer_size[-1]
        for out_channel in reversed(self.hidden_layer_size[:-1]):
            model.append(nn.ConvTranspose1d(prev_ch,out_channel,kernel_size=3,stride=2))
            model.append(nn.ReLU())
            prev_ch = out_channel

        model.append(nn.ConvTranspose1d(prev_ch,self.feat_dim,kernel_size=3,stride=2))
        model.append(nn.ReLU())
        
        model.append(nn.Flatten())
        return nn.Sequential(*model)
    
    def calculate_last_dense_dim(self):
        temp = self.seq_len
        for i in self.hidden_layer_size:
            temp = ((temp - (3-1) - 1) // 2) + 1
        
        return temp * self.hidden_layer_size[-1]


    def forward(self,X): # shape: (B,feats,seq_len)
        z_mean, z_log_var = self.encoding(X)

        z = self.sampling(z_mean,z_log_var)

        _z = self.decoding(z)

        return _z, (z_mean,z_log_var)

    def encoding(self,X):
        _x = self.encoder(X)
        z_mean = self.dense_mean(_x)
        z_log_var = self.dense_var(_x)
        return z_mean, z_log_var

    def decoding(self,z):
        batch_size = z.shape[0]
        _z = self.first_decoder_dense(z)
        _z = torch.reshape(_z,(batch_size,self.hidden_layer_size[-1],-1))
        _z = self.decoder(_z)
        _z = self.last_decoder_dense(_z)
        _z = torch.reshape(_z,(batch_size,self.feat_dim,self.seq_len))
        return _z

    
    def get_prior_samples(self,num_samples):
        Z = torch.randn(num_samples,self.latent_dim)
        _z = self.decoding(Z)
        return _z

    def get_prior_samples_given_Z(self,Z):
        return self.decoding(Z)







        

