import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class AE(nn.Module):
    def __init__(self, input, latent, encode, dropout, decode = None):
        super(AE, self).__init__()
        
        self.input__ = input
        self.dropout__ = dropout
        self.latent__ = latent

        if decode is None:
            decode = encode[::-1]

        self.encode__ = encode            
        encode = [ input ] + encode + [ latent ]

        self.decode__ = decode
        decode = [ latent ] + decode + [ input ]

        # used only for topological loss
        self.latent_norm = torch.nn.Parameter(data=torch.ones(1).cuda(), requires_grad=True)

        self.dropout = nn.Dropout(p = dropout)
        
        self.encode = nn.ModuleList()
        for k in range(len(encode)-1):
            self.encode.append(nn.Linear(encode[k], encode[k+1]))

        self.decode = nn.ModuleList()
        for k in range(len(decode)-1):
            self.decode.append(nn.Linear(decode[k], decode[k+1]))

    def encoder(self, x):
        h = x
        for i in range(len(self.encode)):
            h = self.dropout(h)
            h = F.relu(self.encode[i](h))
        h = self.dropout(h)
        return h

    def decoder(self, z):
        h = z
        for i in range(len(self.decode)-1):
            h = self.dropout(h)
            h = F.relu(self.decode[i](h))

        # last hidden layer
        h = self.dropout(h)
        h = self.decode[-1](h)

        return h
        # return F.sigmoid(h)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

class VAE(nn.Module):
    def __init__(self, input, latent, encode, dropout, decode = None):
        super(VAE, self).__init__()
        
        self.input__ = input
        self.dropout__ = dropout
        self.latent__ = latent

        if decode is None:
            decode = encode[::-1]

        self.encode__ = encode            
        encode = [ input ] + encode + [ latent ]

        self.decode__ = decode
        decode = [ latent ] + decode + [ input ]

        # used only for topological loss
        self.latent_norm = torch.nn.Parameter(data=torch.ones(1).cuda(), requires_grad=True)

        self.dropout = nn.Dropout(p = dropout)
        
        self.encode = nn.ModuleList()
        for k in range(len(encode)-2):
            self.encode.append(nn.Linear(encode[k], encode[k+1]))
        self.encode_mu = nn.Linear(encode[-2], encode[-1])
        self.encode_logvar = nn.Linear(encode[-2], encode[-1])

        self.decode = nn.ModuleList()
        for k in range(len(decode)-1):
            self.decode.append(nn.Linear(decode[k], decode[k+1]))

    def encoder(self, x, sample = False):
        h = x
        for i in range(len(self.encode)):
            h = self.dropout(h)
            h = F.relu(self.encode[i](h))
        h = self.dropout(h)

        mu, logvar = self.encode_mu(h), self.encode_logvar(h)

        if sample == False:
            return mu, logvar
        elif sample == True:
            return self.sampling(mu, logvar)

    def sampling(self, mu, logvar):
        if self.training == True:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu) # return z sample
        else:
            return mu
        
    def decoder(self, z, sample = None):
        h = z
        for i in range(len(self.decode)-1):
            h = self.dropout(h)
            h = F.relu(self.decode[i](h))
        
        # last hidden layer
        h = self.dropout(h)
        h = self.decode[-1](h)
        
        return h
        # return F.sigmoid(h)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.sampling(mu, logvar)
        return self.decoder(z), z, mu, logvar


def save_model(path, model):

    d = {}
    d['dictionary'] = model.state_dict()
    d['input'] = model.input__
    d['latent'] = model.latent__
    d['encode'] = model.encode__
    d['decode'] = model.decode__
    d['dropout'] = model.dropout__

    torch.save(d, path)

def load_model(path):

    d = torch.load(path)
    model = VAE(
        input = d['input'],
        latent = d['latent'],
        encode = d['encode'],
        decode = d['decode'],
        dropout = d['dropout'],
        )
    model.load_state_dict(d['dictionary'])

    return model