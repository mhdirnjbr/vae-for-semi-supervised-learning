import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
import torch.distributions as dist
import os
from data_loader import FashionMNISTCached, FashionMNIST_classes
from architectures import (Classifier, ConditionalPrior,
                       FashionMNISTDecoder, FashionMNISTEncoder)


def compute_kl(locs_q, scale_q, locs_p=None, scale_p=None):
    if locs_p is None:
        locs_p = torch.zeros_like(locs_q)
    if scale_p is None:
        scale_p = torch.ones_like(scale_q)

    dist_q = dist.Normal(locs_q, scale_q)
    dist_p = dist.Normal(locs_p, scale_p)
    return dist.kl.kl_divergence(dist_q, dist_p).sum(dim=-1)

def img_log_likelihood(recon, xs):
        return dist.Laplace(recon, torch.ones_like(recon)).log_prob(xs).sum(dim=(1,2,3))

class CCVAE(nn.Module):
    """
    CCVAE
    """
    def __init__(self, z_dim, num_classes,
                 im_shape, use_cuda, prior_fn):
        super(CCVAE, self).__init__()
        self.z_dim = z_dim
        #self.z_classify = num_classes
        #self.z_style = z_dim - num_classes
        self.z_classify = z_dim
        self.z_style = 0
        self.im_shape = im_shape
        self.use_cuda = use_cuda
        self.num_classes = num_classes
        self.ones = torch.ones(1, self.z_style)
        self.zeros = torch.zeros(1, self.z_style)
        self.y_prior_params = FashionMNISTCached.prior_fn()

        self.encoder = FashionMNISTEncoder(self.z_dim)
        self.decoder = FashionMNISTDecoder(self.z_dim)
        self.classifier = Classifier(z_dim=self.z_dim, classes=self.num_classes)
        self.cond_prior = ConditionalPrior(z_dim=self.z_dim, classes=self.num_classes)

        if self.use_cuda:
            self.ones = self.ones.cuda()
            self.zeros = self.zeros.cuda()
            self.y_prior_params = self.y_prior_params.cuda()
            self.cuda()

    def unsup(self, x):
        bs = x.shape[0]
        #inference
        post_params = self.encoder(x)
        z = dist.Normal(*post_params).rsample()
        zc, zs = z.split([self.z_classify, self.z_style], 1)
        qyzc = dist.Categorical(logits=self.classifier(zc))
        y = qyzc.sample()
        log_qy = qyzc.log_prob(y).sum(dim=-1)

        # compute kl
        y = F.one_hot(y, num_classes=self.num_classes).float()
        locs_p_zc, scales_p_zc = self.cond_prior(y)
        prior_params = (torch.cat([locs_p_zc, self.zeros.expand(bs, -1)], dim=1), 
                        torch.cat([scales_p_zc, self.ones.expand(bs, -1)], dim=1))
        kl = compute_kl(*post_params, *prior_params)

        #compute log probs for x and y
        recon = self.decoder(z)
        log_py = dist.Bernoulli(self.y_prior_params.expand(bs, -1)).log_prob(y).sum(dim=-1)
        elbo = (img_log_likelihood(recon, x) + log_py - kl - log_qy).mean()
        return -elbo
    
    def sup(self, x, y):
        bs = x.shape[0]
        #inference
        post_params = self.encoder(x)
        z = dist.Normal(*post_params).rsample()
        zc, zs = z.split([self.z_classify, self.z_style], 1)
        qyzc = dist.Categorical(logits=self.classifier(zc))
        log_qyzc = qyzc.log_prob(y).sum(dim=-1)

        # compute kl
        y = F.one_hot(y, num_classes=self.num_classes).float()
        locs_p_zc, scales_p_zc = self.cond_prior(y)
        prior_params = (torch.cat([locs_p_zc, self.zeros.expand(bs, -1)], dim=1), 
                        torch.cat([scales_p_zc, self.ones.expand(bs, -1)], dim=1))
        #prior_params = (self.zeros.expand(bs, -1), self.ones.expand(bs, -1))
        kl = compute_kl(*post_params, *prior_params)

        #compute log probs for x and y
        recon = self.decoder(z)
        log_py = dist.Bernoulli(self.y_prior_params.expand(bs, -1)).log_prob(y).sum(dim=-1)
        log_qyx = self.classifier_loss(x, y)
        log_pxz = img_log_likelihood(recon, x)

        # we only want gradients wrt to params of qyz, so stop them propogating to qzx
        log_qyzc_ = dist.Bernoulli(logits=self.classifier(zc.detach())).log_prob(y).sum(dim=-1)
        w = torch.exp(log_qyzc_ - log_qyx)
        elbo = (w * (log_pxz - kl - log_qyzc) + log_py + log_qyx).mean()
        return -elbo

    def classifier_loss(self, x, y, k=100):
        """
        Computes the classifier loss.
        """
        zc, _ = dist.Normal(*self.encoder(x)).rsample(torch.tensor([k])).split([self.z_classify, self.z_style], -1)
        logits = self.classifier(zc.view(-1, self.z_classify))
        d = dist.Bernoulli(logits=logits)
        y = y.expand(k, -1, -1).contiguous().view(-1, self.num_classes)
        lqy_z = d.log_prob(y).view(k, x.shape[0], self.num_classes).sum(dim=-1)
        lqy_x = torch.logsumexp(lqy_z, dim=0) - np.log(k)
        return lqy_x

    def reconstruct_img(self, x):
        return self.decoder(dist.Normal(*self.encoder(x)).rsample())

    def classifier_acc(self, x, y=None, k=1):
        zc, _ = dist.Normal(*self.encoder(x)).rsample(torch.tensor([k])).split([self.z_classify, self.z_style], -1)
        logits = self.classifier(zc.view(-1, self.z_classify)).view(-1, self.num_classes)
        #y = y.expand(k, -1,-1).contiguous().view(-1, self.num_classes)
        #y = y.expand(k, -1).contiguous()
        preds = torch.softmax(logits, dim=1)
        #acc = (preds.eq(y)).float().mean()
        acc = (torch.max(preds,dim=1).indices.eq(y)).float().mean()
        return acc
    
    def save_models(self, path='./data'):
        torch.save(self.encoder, os.path.join(path,'encoder.pt'))
        torch.save(self.decoder, os.path.join(path,'decoder.pt'))
        torch.save(self.classifier, os.path.join(path,'classifier.pt'))
        torch.save(self.cond_prior, os.path.join(path,'cond_prior.pt'))

    def accuracy(self, data_loader, *args, **kwargs):
        acc = 0.0
        for (x, y) in data_loader:
            if self.use_cuda:
                x, y = x.cuda(), y.cuda()
            batch_acc = self.classifier_acc(x, y)
            acc += batch_acc
        return acc / len(data_loader)
    
    def latent_walk(self, images, save_dir):
        #img_1=images[0]
        img_2=images[3]
        img_3=images[20]
        #img_4=images[40]
        latent_2=dist.Normal(*self.encoder(img_2)).sample()
        latent_3=dist.Normal(*self.encoder(img_3)).sample()
        num_points = 8
        interpolated_points = [torch.lerp(latent_2, latent_3, i/(num_points+1)) for i in range(1, num_points+1)]
        img_recon=[]
        for point in interpolated_points:
            img_recon.append(torch.squeeze(self.decoder(point).view(-1, *self.im_shape),dim=1))
        grid_recon = make_grid(img_recon)
        save_image(grid_recon, os.path.join(save_dir, "test.png"))
        
    def latent_traversal(self, images, save_dir):
        # Select four images for interpolation
        img_1 = images[0]
        img_2 = images[9]
        img_3 = images[11]
        img_4 = images[41]

        # Encode the selected images to obtain latent vectors
        latent_1 = dist.Normal(*self.encoder(img_1)).sample()
        latent_2 = dist.Normal(*self.encoder(img_2)).sample()
        latent_3 = dist.Normal(*self.encoder(img_3)).sample()
        latent_4 = dist.Normal(*self.encoder(img_4)).sample()

        # Perform interpolation between latent vectors for all four corners
        num_points = 8
        interpolated_latents = []
        for i in range(num_points):
            for j in range(num_points):
                alpha = i / (num_points - 1)
                beta = j / (num_points - 1)
                
                interpolated_latent = (
                    (1 - alpha) * (1 - beta) * latent_1 +
                    alpha * (1 - beta) * latent_2 +
                    (1 - alpha) * beta * latent_3 +
                    alpha * beta * latent_4
                )
                interpolated_latents.append(interpolated_latent)

        # Decode interpolated latent vectors and reconstruct images
        img_recon=[]
        for point in interpolated_latents:
            img_recon.append(torch.squeeze(self.decoder(point).view(-1, *self.im_shape),dim=1))

        # Create a grid of reconstructed images and save it
        grid_recon = make_grid(img_recon, nrow=num_points)
        save_image(grid_recon, os.path.join(save_dir, "interpolation_result_bis.png"))