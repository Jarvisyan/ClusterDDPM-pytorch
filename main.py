import os
import random
import math
from typing import Dict, Tuple
from tqdm import tqdm
import torch
import itertools
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import models, transforms

from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import ipdb
from sklearn.mixture import GaussianMixture
from sklearn.metrics.cluster import normalized_mutual_info_score as NMI

from torch.optim.lr_scheduler import StepLR


class ResidualConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        '''
        standard ResNet style convolutional block
        '''
        self.same_channels = in_channels==out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            # this adds on correct residual in case channels have increased
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2 
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2
        
class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        '''
        process and downscale the image feature maps
        '''
        layers = [ResidualConvBlock(in_channels, out_channels), nn.MaxPool2d(2)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        '''
        process and upscale the image feature maps
        '''
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1) ## channels cat
        x = self.model(x)
        return x

class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super().__init__()
        '''
        generic one layer FC NN for embedding things  
        '''
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)

class ContextUnet(nn.Module):
    def __init__(self, in_channels = 1, n_feat = 128, hid_dim = 10):
        # context_dim = 10 if class condition, 2 * n_feat if semantic condition
        super().__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.context_dim = hid_dim

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True) ##1 * 28 * 28 -> n_feat * 28 * 28 -> n_feat * 28 * 28

        self.down1 = UnetDown(n_feat, n_feat) ##n_feat * 28 * 28 -> n_feat * 28 * 28 -> n_feat * 14 * 14 
        self.down2 = UnetDown(n_feat, 2 * n_feat) ##n_feat * 14 * 14 -> 2 * n_feat * 14 * 14 -> 2 * n_feat * 7 * 7 

        self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU()) ## 2 * n_feat * 7 * 7 -> 2 * n_feat * 1 * 1

        self.timeembed1 = EmbedFC(1, 2*n_feat) ## 1 -> 2 * n_feat -> 2 * n_feat
        self.timeembed2 = EmbedFC(1, 1*n_feat)
        self.contextembed1 = EmbedFC(self.context_dim, 2*n_feat) ## 10 -> 2 * n_feat -> 2 * n_feat
        self.contextembed2 = EmbedFC(self.context_dim, 1*n_feat)

        self.up0 = nn.Sequential(
            # nn.ConvTranspose2d(6 * n_feat, 2 * n_feat, 7, 7), # when concat temb and cemb end up w 6*n_feat
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 7, 7), # otherwise just have 2*n_feat: ## 2 * n_feat * 1 * 1 -> 2 * n_feat * 7 * 7
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(4 * n_feat, n_feat) ## 4 * n_feat * 7 * 7 -> n_feat * 14 * 14 -> n_feat * 14 * 14 -> n_feat * 14 * 14 -> n_feat * 14 * 14 -> n_feat * 14 * 14
        self.up2 = UnetUp(2 * n_feat, n_feat) ## 2 * n_feat * 14 * 14 -> n_feat * 28 * 28 -> n_feat * 28 * 28 -> n_feat * 28 * 28 -> n_feat * 28 * 28 -> n_feat * 28 * 28
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )  ## 2 * n_feat * 28 * 28 ->  n_feat * 28 * 28 ->  1 * 28 * 28

    def forward(self, x, c, t):
        # x is (noisy) image, c is context, t is timestep, 
        # context_mask says which samples to block the context on

        x = self.init_conv(x)  ## 1 * 28 * 28 -> n_feat * 28 * 28 -> n_feat * 28 * 28
        down1 = self.down1(x)  ## n_feat * 28 * 28 -> n_feat * 28 * 28 -> n_feat * 14 * 14 
        down2 = self.down2(down1) ## n_feat * 14 * 14 -> 2 * n_feat * 14 * 14 -> 2 * n_feat * 7 * 7
        hiddenvec = self.to_vec(down2) ## 2 * n_feat * 7 * 7 -> 2 * n_feat * 1 * 1
        
        # embed context, time step
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)## self.context_dim -> 2 * n_feat -> 2 * n_feat -> 2 * n_feat * 1 * 1
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)   ## 1                -> 2 * n_feat -> 2 * n_feat -> 2 * n_feat * 1 * 1
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)    ## self.context_dim ->     n_feat ->     n_feat ->     n_feat * 1 * 1
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        # could concatenate the context embedding here instead of adaGN
        # hiddenvec = torch.cat((hiddenvec, temb1, cemb1), 1)

        up1 = self.up0(hiddenvec)  ## 2 * n_feat * 1 * 1 -> 2 * n_feat * 7 * 7
        # up2 = self.up1(up1, down2) # if want to avoid add and multiply embeddings
        up2 = self.up1(cemb1*up1+ temb1, down2)  # add and multiply embeddings: 
        ## 4 * n_feat * 7 * 7 -> n_feat * 14 * 14 -> n_feat * 14 * 14 -> n_feat * 14 * 14 -> n_feat * 14 * 14 -> n_feat * 14 * 14
        up3 = self.up2(cemb2*up2+ temb2, down1)
        ## 2 * n_feat * 14 * 14 -> n_feat * 28 * 28 -> n_feat * 28 * 28 -> n_feat * 28 * 28 -> n_feat * 28 * 28 -> n_feat * 28 * 28
        out = self.out(torch.cat((up3, x), 1)) ## 2 * n_feat * 28 * 28 ->  n_feat * 28 * 28 -> 1 * 28 * 28
        return out
    
class Encoder(nn.Module):
    def __init__(self, in_channels = 1, n_feat = 128, hid_dim = 10):
        super().__init__()
        self.in_channels = in_channels
        self.n_feat = n_feat
        
        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True) ##1 * 28 * 28 -> n_feat * 28 * 28 -> n_feat * 28 * 28
        self.down1 = UnetDown(n_feat, n_feat) ##n_feat * 28 * 28 -> n_feat * 28 * 28 -> n_feat * 14 * 14 
        self.down2 = UnetDown(n_feat, 2 * n_feat) ##n_feat * 14 * 14 -> 2 * n_feat * 14 * 14 -> 2 * n_feat * 7 * 7 

        # self.mu_l = nn.Linear(2 * n_feat * 7 * 7, hid_dim)
        # self.log_sigma2_l = nn.Linear(2 * n_feat * 7 * 7, hid_dim)  
        self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU()) ## 2 * n_feat * 7 * 7 -> 2 * n_feat * 1 * 1 -> 2 * n_feat
        self.mu_l = nn.Linear(2 * n_feat, hid_dim)
        self.log_sigma2_l = nn.Linear(2 * n_feat, hid_dim)      
    def forward(self, x):
        x = self.init_conv(x)  ## 1 * 28 * 28 -> n_feat * 28 * 28 -> n_feat * 28 * 28
        down1 = self.down1(x)  ## n_feat * 28 * 28 -> n_feat * 28 * 28 -> n_feat * 14 * 14 
        down2 = self.down2(down1) ## n_feat * 14 * 14 -> 2 * n_feat * 14 * 14 -> 2 * n_feat * 7 * 7
        
        # hiddenvec = down2.view(-1, 2 * self.n_feat * 7 * 7)
        hiddenvec = self.to_vec(down2).squeeze_() ## 2 * n_feat * 7 * 7 -> 2 * n_feat * 1 * 1 -> 2 * n_feat   
        return self.mu_l(hiddenvec), self.log_sigma2_l(hiddenvec)

def ddpm_schedules(beta1, beta2, T):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1 ##torch.linspace(beta1, beta2, T + 1)
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


class Context_DDPM(nn.Module):
    def __init__(self, cunet, encoder, nClusters, hid_dim, beta1, beta2, n_T, device, drop_prob = 0.1):
        super().__init__()
        self.cunet = cunet
        self.encoder = encoder
        
        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddpm_schedules(beta1, beta2, n_T).items():
            self.register_buffer(k, v)
        
        self.nClusters = nClusters
        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()
    
    def pre_train(self, dataloader, lr_pre = 1e-4, pre_epoch = 10, vis_dir = './'): 
        if  not os.path.exists('./last.pt'):
            optim = torch.optim.Adam(itertools.chain(self.encoder.parameters(), self.cunet.parameters()), lr = lr_pre)#, weight_decay = 1e-5)
            print('Pretraining......')
            for ep in range(pre_epoch):
                print(f'epoch {ep}:')
                self.encoder.train()
                self.cunet.train()
                # cosine lrate decay
                #adjust_learning_rate(optim, ep, pre_epoch, lr_pre)
                
                pbar = tqdm(dataloader)
                loss_ema = None
                for x, _ in pbar:
                    optim.zero_grad()
                    
                    x = x.to(self.device)
                    loss = self.Rec_loss(x, pretrain = True)
                    loss.backward()
                    if loss_ema is None:
                        loss_ema = loss.item()
                    else:
                        loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
                    pbar.set_description(f"loss: {loss_ema:.4f}")
                    optim.step()
                
                gmm, acc, nmi = self.predict(dataloader)
                
                save_dir = './outputs/mnist/'
                # for eval, save an image of currently generated samples (top rows)
                # followed by real images (bottom rows)
                self.encoder.eval()
                self.cunet.eval()
                with torch.no_grad():
                    n_sample = 10
                    x_real = x[:n_sample]
                    w = 0.
                    x_gen, _ = self.sample(x_real, self.device, guide_w = w)
                    rec_loss = self.loss_mse(x_gen, x_real)
                    print(f'Rec_loss: = {rec_loss : .4f}')
                    x_all = torch.cat([x_gen, x_real])
                    grid = make_grid(x_all, nrow=10) 
                    save_dir = os.path.join(vis_dir, f"image_ep{ep}_w{w}.png")
                    save_image(grid, save_dir)
                    print('saved image at ' + save_dir)
            
            state = {
                'state_dict' : self.state_dict(),
                'gmm' : gmm,
                'acc' : acc
            }
            torch.save(state, './last.pt')            
            return gmm
        
        else:
            checkpoint = torch.load('./last.pt') #
            self.load_state_dict(checkpoint['state_dict'])
            gmm = checkpoint['gmm']            
            return gmm
        
        
    def Rec_loss(self, x, pretrain = False): ##pretrain = True -> diffAE; 
        
        """
        this method is used in training, so samples t and noise randomly
        """
        #--------------- noise reconstruction loss---------------
        _ts = torch.randint(1, self.n_T + 1, (x.shape[0], )).to(self.device)  # t ~ Uniform(0, n_T)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)
        
        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * noise
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.
    
        z_mu, z_sigma2_log = self.encoder(x)
        z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu

        # dropout context with some probability
        context_mask = torch.bernoulli(torch.zeros(z.shape[0]) + self.drop_prob).to(self.device)
        context_mask = context_mask[:, None]
        context_mask = context_mask.repeat(1, z.shape[1])
        #ipdb.set_trace()

        # mask out context if context_mask == 1
        context_mask = 1 - context_mask
        z = z * context_mask
        if pretrain: 
            return self.loss_mse(noise, self.cunet(x_t, z, _ts / self.n_T))
        else:
            return nn.MSELoss(reduction = 'none')(noise, self.cunet(x_t, z, _ts / self.n_T)).view(x.shape[0], -1).sum(1).mean()
        #return nn.MSELoss(reduction = 'none')(noise, self.cunet(x_t, z, _ts / self.n_T)).view(x.shape[0], -1).sum(1).mean() #self.loss_mse(noise, self.cunet(x_t, z, _ts / self.n_T)) #
    
    def predict(self, dataloader, gmm = None):
        self.encoder.eval()
        latent_z = []
        ground_truth = []
        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(self.device)
                z_mu, z_sigma2_log = self.encoder(x)
                z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu
                latent_z.append(z)
                ground_truth.append(y)
        latent_z = torch.cat(latent_z, 0).detach().cpu().numpy()
        ground_truth = torch.cat(ground_truth, 0).detach().numpy()
        
        if gmm is not None:
            init_weights = gmm.weights_
            init_means = gmm.means_
            init_precisions = gmm.precisions_
            gmm = GaussianMixture(n_components = self.nClusters, covariance_type = 'diag', 
                                weights_init = init_weights, means_init = init_means, precisions_init = init_precisions)
            print('With initial:')
        else:
            gmm = GaussianMixture(n_components = self.nClusters, covariance_type = 'diag')
            print('No initial')
        pre = gmm.fit_predict(latent_z)
        acc = cluster_acc(pre, ground_truth)[0]
        nmi = NMI(pre, ground_truth)
        print(f'Acc = {acc * 100 : .4f}, NMI = {nmi : .4f} \n')
        return gmm, acc, nmi
        
#         c_mu = torch.from_numpy(gmm.means_).to(x).float()
#         c_sigma2_log = torch.log(torch.from_numpy(gmm.covariances_).to(x).float())
#         pred_prob = torch.from_numpy(gmm.predict_proba(latent_z)).to(x).float()
#         return c_mu, c_sigma2_log, pred_prob, acc
    
    def ELBO_Loss(self, x, gmm):
        
        """
        this method is used in training, so samples t and noise randomly
        """
        #--------------- noise reconstruction loss---------------
        rec_loss = self.Rec_loss(x) #* 40
        
        
        #----------------KL loss---------------
        pi = torch.from_numpy(gmm.weights_).to(x).float()
        c_mu = torch.from_numpy(gmm.means_).to(x).float()
        c_sigma2_log = torch.log(torch.from_numpy(gmm.covariances_).to(x).float())

        z_mu, z_sigma2_log = self.encoder(x)
        z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu
        
        det = 1e-10
        yita_c = torch.exp(torch.log(pi.unsqueeze(0)) + self.gaussian_pdfs_log(z, c_mu, c_sigma2_log)) + det
        yita_c = yita_c / (yita_c.sum(1).view(-1,1))#batch_size*Clusters
        

        
        # pred_prob = torch.from_numpy(gmm.predict_proba(z.detach().cpu().numpy())).to(x).float()        
        
        ###################################################
        kl_loss = 0.5 * torch.mean(torch.sum(yita_c * torch.sum(c_sigma2_log.unsqueeze(0) + 
                                                              torch.exp(z_sigma2_log.unsqueeze(1) - c_sigma2_log.unsqueeze(0)) + 
                                                              (z_mu.unsqueeze(1) - c_mu.unsqueeze(0)).pow(2) / torch.exp(c_sigma2_log.unsqueeze(0)), 2), 1)) 
        ## 1 * C * J,  
        ## N * 1 * J, 1 * C * J, 
        ## N * 1 * J, 1 * C * J, 1 * C * J
        
        
        kl_loss -= torch.mean(torch.sum(yita_c * torch.log(pi.unsqueeze(0) / (yita_c + det)), 1)) + 0.5 * torch.mean(torch.sum(1 + z_sigma2_log, 1))        
        return rec_loss, kl_loss, (rec_loss / kl_loss ).detach()

    def gaussian_pdfs_log(self,x,mus,log_sigma2s):
        G=[]
        for c in range(self.nClusters):
            G.append(self.gaussian_pdf_log(x,mus[c:c+1,:],log_sigma2s[c:c+1,:]).view(-1,1)) ##batch_size*1
        return torch.cat(G,1) ##batch_size * num_clusters




    @staticmethod
    def gaussian_pdf_log(x,mu,log_sigma2):
        return -0.5*(torch.sum(np.log(np.pi*2)+log_sigma2+(x-mu).pow(2)/torch.exp(log_sigma2),1))


    def sample(self, x, device, guide_w = 0.0):
        # we follow the guidance sampling scheme described in 'Classifier-Free Diffusion Guidance'
        # to make the fwd passes efficient, we concat two versions of the dataset,
        # one with context_mask=0 and the other context_mask=1
        # we then mix the outputs with the guidance scale, w
        # where w>0 means more guidance
        
        n_sample = x.shape[0]
        x_i = torch.randn_like(x)  # x_T ~ N(0, 1), sample initial noise
        
        z_mu, z_sigma2_log = self.encoder(x)
        z_i = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu

        # don't drop context at test time
        context_mask = torch.zeros_like(z_i)

        # double the batch
        z_i = z_i.repeat(2, 1) 
        context_mask = context_mask.repeat(2, 1)
        context_mask[n_sample:] = 1. # makes second half of batch context free
        # mask out context if context_mask == 1
        context_mask = 1 - context_mask
        z_i = z_i * context_mask   

        x_i_store = [] # keep track of generated steps in case want to plot something 
        print()
        for i in range(self.n_T, 0, -1):
            print(f'sampling timestep {i}',end='\r')
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample,1,1,1)

            # double batch
            x_i = x_i.repeat(2,1,1,1)
            t_is = t_is.repeat(2,1,1,1)

            z = torch.randn_like(x) if i > 1 else 0

            # split predictions and compute weighting
            eps = self.cunet(x_i, z_i, t_is)
            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]
            eps = (1+guide_w)*eps1 - guide_w*eps2
            x_i = x_i[:n_sample]
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
            if i%20==0 or i==self.n_T or i<8:
                x_i_store.append(x_i.detach().cpu().numpy())
        
        x_i_store = np.array(x_i_store)
        return x_i, x_i_store    
        
def cluster_acc(Y_pred, Y):
    #from sklearn.utils.linear_assignment_ import linear_assignment
    from scipy.optimize import linear_sum_assignment
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max())+1
    w = np.zeros((D,D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_sum_assignment(w.max() - w) #linear_assignment(w.max() - w)
    return sum([w[i,j] for i,j in zip(ind[0], ind[1])])*1.0/Y_pred.size, w #sum([w[i,j] for i,j in ind])*1.0/Y_pred.size, w

def adjust_learning_rate(optimizer, epoch_now, epoch_total, lr):
    """Decay the learning rate based on schedule"""
    # cosine lr schedule
    lr *= 0.5 * (1. + math.cos(math.pi * epoch_now / epoch_total))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True 

def train_mnist():
    seed = 2  #2, 7
    setup_seed(seed)

    in_channels = 1
    n_feat = 64 
    hid_dim = 32 
    
    nClusters = 10
    beta1 = 1e-4
    beta2 = 0.02
    n_T = 400 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    drop_prob = 0.1
    
    batch_size = 128 
    lr_pre = 1e-4
    pre_epoch = 10 
    
    lr_train = 1e-4 
    n_epoch = 570
    save_dir = './outputs/mnist/'
    ws_test = [0.0]#, 0.5, 2.0] # strength of generative guidance
    
    save_every_epoch = 10
    model_dir = os.path.join(save_dir, "ckpts")
    vis_dir = os.path.join(save_dir, "visual")
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    load_epoch = -1
    
    cunet = ContextUnet(in_channels = in_channels, n_feat = n_feat, hid_dim = hid_dim).to(device)
    encoder = Encoder(in_channels = in_channels, n_feat = n_feat, hid_dim = hid_dim).to(device)

    train_dataset = MNIST(root = '../data', train = True, download = True, transform = transforms.ToTensor())
    test_dataset = MNIST(root = '../data', train = False, download = True, transform = transforms.ToTensor())
    combined_dataset = ConcatDataset([train_dataset, test_dataset])    
    dataloader = DataLoader(combined_dataset, batch_size, shuffle = True, num_workers = 5) 
    
    if load_epoch == -1:  ##pretraining
        cddpm = Context_DDPM(cunet, encoder, nClusters, hid_dim, beta1, beta2, n_T, device, drop_prob)
        cddpm.to(device)
        gmm = cddpm.pre_train(dataloader, lr_pre = lr_pre, pre_epoch = pre_epoch, vis_dir = vis_dir)
        
        optim = torch.optim.Adam(cddpm.parameters(), lr = lr_train)
        #lr_s = StepLR(optim, step_size = 10, gamma = 0.95)
    else: ##resume
        target = os.path.join(model_dir, f"model_{load_epoch}.pth")
        print("loading model at", target)
        checkpoint = torch.load(target, map_location=device)
        cunet.load_state_dict(checkpoint['cunet'])
        encoder.load_state_dict(checkpoint['encoder'])
        
        cddpm = Context_DDPM(cunet, encoder, nClusters, hid_dim, beta1, beta2, n_T, device, drop_prob)
        cddpm.to(device)
        optim = torch.optim.Adam(cddpm.parameters(), lr = lr_train)
        optim.load_state_dict(checkpoint['opt'])
        gmm = checkpoint['gmm']

    for ep in range(load_epoch + 1, n_epoch):
        print(f'epoch {ep}')
        cddpm.train()
        
        # cosine lrate decay
        #adjust_learning_rate(optim, ep, n_epoch, lr_train)
        #------------------------------M-step-------------------------------
        ##mu_c, sigma_c, pred_prob
        
        print(f'M-step:.......')
        # for g in optim.param_groups:
        #     g['lr'] = lr_train * min((ep + 1.0) / 5, 1.0) # warmup         
        now_lr = optim.param_groups[0]['lr']
        print(f'epoch {ep}, lr {now_lr:f}')
        
        pbar = tqdm(dataloader)
        loss_ema_rec = None
        loss_ema_kl = None
        for x, _ in pbar:
            optim.zero_grad()

            x = x.to(device)
            rec_loss, kl_loss, frac = cddpm.ELBO_Loss(x, gmm)
            kl_loss *= 0.1
            loss_all = rec_loss + kl_loss
            loss_all.backward()
            if loss_ema_rec is None:
                loss_ema_rec = rec_loss.item()
                loss_ema_kl = kl_loss.item()
            else:
                loss_ema_rec = 0.95 * loss_ema_rec + 0.05 * rec_loss.item()
                loss_ema_kl = 0.95 * loss_ema_kl + 0.05 * kl_loss.item()
            pbar.set_description(f"Rec_loss: {loss_ema_rec : .4f}, KL_loss:{loss_ema_kl : .4f}, Frac : {frac : .4f}")
            optim.step()
        #lr_s.step()
        
        #-----------------------------E-step--------------------------------
        print('E-step:.......')
        gmm, acc, nmi = cddpm.predict(dataloader, gmm)
        
        with open(os.path.join(save_dir, "loss_v3.txt"), 'a') as f:
            f.write(f'Seed: {seed} | Epoch: {ep : 03d} | Rec_loss: {loss_ema_rec : .4f} | KL_loss:{loss_ema_kl : .4f} | Frac : {frac : .4f} | ACC = {acc : .4f} | NMI = {nmi : .4f}\n')
        
        # save model
        if (ep % save_every_epoch == 0) or (ep == (n_epoch - 1)):
            checkpoint = {
                'cunet': cddpm.cunet.state_dict(),
                'encoder': cddpm.encoder.state_dict(),
                'opt': optim.state_dict(),
                'gmm' : gmm,
                'acc' : acc,
                'nmi' : nmi
            }
            save_path = os.path.join(model_dir, f"model_{ep}.pth")
            torch.save(checkpoint, save_path)
            print('saved model at', save_path)



if __name__ == "__main__":
    train_mnist()
    
#python main.py