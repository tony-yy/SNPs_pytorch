import os
import argparse
import random
import numpy as np

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F

class SNP(nn.Module):
    def __init__(self, args):
        super(SNP, self).__init__()
        # hyperparams
        self._beta = args.beta
        self.num_latents = args.num_latents
        self.latent_encoder_output_sizes = args.latent_encoder_output_sizes
        self.deterministic_encoder_output_sizes=args.deterministic_encoder_output_sizes
        # an important one: size of global representation from encoder
        self.representation_size = args.representation_size
        # above; to be determined
        self.decoder_output_sizes = args.decoder_output_sizes
                
        # latent encoder layers
        # 1-d dataset, input: [batch_size, (x_cxt, y_cxt)] 
        self.lat_enc_1 = nn.Linear(2, 128)
        self.lat_enc_2 = nn.Linear(128, 128)
        self.lat_enc_3 = nn.Linear(128, 128)
        # after adding vrnn_h to latent_encoding
        self.lat_enc_4 = nn.Linear(128, 128)
        self.lat_enc_5 = nn.Linear(128, 128)

        # deterministic encoder layers
        self.det_enc_1 = nn.Linear(2, 128)
        self.det_enc_2 = nn.Linear(128, 128)
        self.det_enc_3 = nn.Linear(128, self.deterministic_encoder_output_sizes[-1])

        # latent_encoding sampling layers
        self.r_to_z_mean = nn.Linear(self.latent_encoder_output_sizes[-1], self.num_latents)
        self.r_to_z_log_sigma = nn.Linear(self.latent_encoder_output_sizes[-1], self.num_latents)

        # decoder layers
        self.dec_1 = nn.Linear(self.representation_size + 1, 128)
        self.dec_2 = nn.Linear(128, 2)   

        # recurrent layers
        self._drnn = nn.LSTMCell(input_size=128, hidden_size=self.num_latents)
        self._vrnn = nn.LSTMCell(input_size=128, hidden_size=self.num_latents)

        # unseen/seen plots
        self._index = torch.Tensor(np.arange(0,2000)).to(torch.device("cuda"))

        # attention layers
        # TBD

    def reparameterise(self, z, num_trgt_t):
        mu, logvar = z
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z_sample = eps.mul(std).add_(mu)
        z_sample = z_sample.unsqueeze(1).expand(-1, num_trgt_t, -1)
        return z_sample

    def latent_encode(self, x, y, vrnn_h, num_cxt_pnt=0, get_hidden=False, given_hidden=None):
        if given_hidden is None: 
            x_y = torch.cat([x, y], dim=-1)
            hidden = F.relu(self.lat_enc_1(x_y))
            hidden = F.relu(self.lat_enc_2(hidden))
            hidden = F.relu(self.lat_enc_3(hidden))
            # last layer without ReLU, supplemented
            hidden = self.lat_enc_5(hidden)
            # hidden = torch.nn.ModuleList(hidden)
        
            if get_hidden:
                return hidden
        else:
            hidden = given_hidden
        
        # aggregator
        hidden = torch.mean(hidden, dim=1)
        hidden = torch.zeros(x.shape[0], self.num_latents).to(torch.device("cuda")) if num_cxt_pnt==0 else hidden
        # incorporate temporal state
        hidden += vrnn_h

        hidden = F.relu(self.lat_enc_4(hidden))
        mu = self.r_to_z_mean(hidden)
        # mu = torch.nn.ModuleList(mu)
        log_sigma = self.r_to_z_log_sigma(hidden)
        # log_sigma = torch.nn.ModuleList(log_sigma)

        return mu, log_sigma
    
    def deterministic_encode(self, cxt_x, cxt_y, drnn_h, num_trgts, num_cxt_pnt=0, get_hidden=False, given_hidden=None, attention=None, trgt_x=None):
        # the argument trgt_x is obsolete
        if given_hidden is None:
            x_y = torch.cat([cxt_x, cxt_y], dim=-1)
            if x_y.shape[1] == 0:
                assert num_cxt_pnt==0, "inconsistent context point number"
            hidden = F.relu(self.det_enc_1(x_y))
            hidden = F.relu(self.det_enc_2(hidden))
            hidden = F.relu(self.det_enc_3(hidden))
            # hidden = torch.nn.ModuleList(hidden)
            if get_hidden:
                # return raw hidden tensor, without temporal information
                return hidden
        else:
            hidden = given_hidden
        # incorporate temporal state
        hidden = torch.mean(hidden, dim=1, keepdim=True).repeat([1, num_trgts, 1])
        drnn_h = torch.unsqueeze(drnn_h, axis=1).repeat([1, num_trgts, 1])
        hidden = drnn_h if num_cxt_pnt==0 else hidden+drnn_h

        return hidden
    
    def decode(self, representation, trgt_x):
        hidden = torch.cat([representation, trgt_x], dim=-1)
        hidden = F.relu(self.dec_1(hidden))
        hidden = self.dec_2(hidden)
        # get mean and variance
        # print(hidden.shape)
        mu, log_sigma = torch.split(hidden, 1, dim=-1)
        # bound variance
        sigma = 0.1 + 0.9 * torch.nn.Softplus()(log_sigma)
        # get the distribution
        dist = torch.distributions.Normal(loc=mu, scale=sigma)

        return dist, mu, sigma

    def forward(self, query, trgt_y, num_trgts, num_cxt_pnts):
        (cxt_x, cxt_y), trgt_x = query
        len_seq = len(cxt_x)
        batch_size = cxt_x[0].shape[0]
        
        # latent rnn state initialization
        lat_h_0 = torch.zeros(batch_size, self.num_latents).to(torch.device("cuda"))
        lat_c_0 = torch.zeros(batch_size, self.num_latents).to(torch.device("cuda"))
        vrnn_h, vrnn_c = lat_h_0, lat_c_0
        latent_rep = vrnn_h

        # deterministic rnn state initialization
        det_h_0 = torch.zeros(batch_size, self.num_latents).to(torch.device("cuda"))
        det_c_0 = torch.zeros(batch_size, self.num_latents).to(torch.device("cuda"))
        drnn_h, drnn_c = det_h_0, det_c_0
        avg_det_rep = drnn_h

        # loss components
        trgt_dist_list, trgt_mu_list, trgt_sigma_list = [], [], []
        trgt_mu_n_sigma = {"mu":[], "sigma":[]}
        # encoder distribution list
        prior_dist = {"mu":[], "log_sigma":[]}
        post_dist = {"mu":[], "log_sigma":[]}

        # recurrent loop
        for t in range(len_seq):
            cxt_x_t = cxt_x[t]
            cxt_y_t = cxt_y[t]
            trgt_x_t = trgt_x[t]
            trgt_y_t = trgt_y[t]
            
            # 1.1 latent encoding
            lat_en_c = self.latent_encode(cxt_x_t, cxt_y_t, vrnn_h, get_hidden=True)
            lat_en_c_mean = torch.mean(lat_en_c, dim=1)
            latent_rep = latent_rep if num_cxt_pnts[t] == 0 else latent_rep + lat_en_c_mean
            vrnn_h, vrnn_c = self._vrnn(latent_rep, (vrnn_h, vrnn_c))
            # pull these out of the if-else clause for loss components calculation
            post_mu, post_log_sigma = self.latent_encode(trgt_x_t, trgt_y_t, vrnn_h, num_cxt_pnts[t])
            prior_mu, prior_log_sigma = self.latent_encode(cxt_x_t, cxt_y_t, vrnn_h, num_cxt_pnts[t], given_hidden=lat_en_c)

            if self.training:
                # assert trgt_y, "tar_y is empty"
                latent_rep = self.reparameterise((post_mu, post_log_sigma), num_trgts[t])
            else:
                latent_rep = self.reparameterise((prior_mu, prior_log_sigma), num_trgts[t])

            # 1.2 deterministic encoding
            det_en_c = self.deterministic_encode(cxt_x_t, cxt_y_t, drnn_h, num_trgts[t], num_cxt_pnts[t], get_hidden=True)
            det_en_c_mean = torch.mean(det_en_c, dim=1)
            avg_det_rep = avg_det_rep if num_cxt_pnts[t] == 0 else avg_det_rep + det_en_c_mean
            drnn_h, drnn_c = self._drnn(avg_det_rep, (drnn_h, drnn_c))
            deterministic_rep = self.deterministic_encode(cxt_x_t, cxt_y_t, drnn_h, num_trgts[t], num_cxt_pnts[t], given_hidden=det_en_c)
            # avg_det_rep = torch.mean(deterministic_rep, dim=1)

            # 1.3 representation merging
            representation = torch.cat([deterministic_rep, latent_rep], dim=-1)
            latent_rep = torch.mean(latent_rep, dim=1)

            # 2.1 decoding
            dist, mu, sigma = self.decode(representation, trgt_x_t)
            
            # output sort-out
            trgt_dist_list.append(dist)
            trgt_mu_n_sigma["mu"].append(mu)
            trgt_mu_n_sigma["sigma"].append(sigma)
            
            prior_dist["mu"].append(prior_mu)
            prior_dist["log_sigma"].append(prior_log_sigma)
            post_dist["mu"].append(post_mu)
            post_dist["log_sigma"].append(post_log_sigma)

            # 2.2. calculating loss components and other metrics
            '''
            if self.training:
                assert trgt_y, "no training label" 
                # 2.2.1 ELBO/log_prob | TrgtNLL/NLL
                log_p = -torch.mean(dist.log_prob(trgt_y_t))
                log_p_list.append(log_p)
                # 2.2.2 ELBO/kl
                kl = -0.5*(-prior_log_sigma+post_log_sigma \
                            - (torch.exp(post_log_sigma)+(post_mu-prior_mu)**2) \
                            / torch.exp(prior_log_sigma) + 1.0)
                kl = torch.mean(kl, dim=-1, keepdims=True)
                kl_list.append(torch.mean(kl))
                # 2.2.3 TrgtNLL/NLLWithoutCxt
                log_p_wo_cxt += log_p if num_cxt_pnts[t]==0 else torch.Tensor([0.0])
                cxt_wo += torch.Tensor([1.0]) if num_cxt_pnts[t]==0 else torch.Tensor([0.0])
                # 2.2.4 TrgtNLL/NLLWithCxt
                log_p_w_cxt += torch.Tensor([0.0]) if num_cxt_pnts[t]==0 else log_p
                cxt_w += torch.Tensor([0.0]) if num_cxt_pnts[t]==0 else torch.Tensor([1.0])
                # 2.2.5 TrgtMSE/MSE
                mse = torch.mean( nn.MSELoss(trgt_y_t, mu) )
                mse_list.append(mse)
                # 2.2.6 TrgtMSE/MSEWithCxt
                mse_w_cxt += torch.Tensor([0.0]) if num_cxt_pnts[t]==0 else mse
                # 2.2.7 TrgtMSE/MSEWithoutCxt
                mse_wo_cxt += mse if num_cxt_pnts[t]==0 else torch.Tensor([0.0])
            '''
        return trgt_dist_list, trgt_mu_n_sigma, prior_dist, post_dist
    
    def snp_loss(self, trgt_dist_list, trgt_mu_n_sigma, prior_dists, post_dists, len_seq, trgt_ys):
        assert trgt_ys, "no training label" 
        # print("len_seq:{}, trgt_dist_list:{}, trgt_mu_:{}, trgt_sigma_:{}, prior_dists_mu:{}, prior_dists_sigma:{}, post_dists_mu:{}, post_dists_sigma:{}, trgt_ys:{}" .format(
        #     len_seq, len(trgt_dist_list), 
        #     len(trgt_mu_n_sigma["mu"]), len(trgt_mu_n_sigma["sigma"]), 
        #     len(prior_dists["mu"]), len(prior_dists["log_sigma"]), 
        #     len(post_dists["mu"]), len(post_dists["log_sigma"]), 
        #     len(trgt_ys)))
        assert len_seq==len(trgt_dist_list) \
                    ==len(trgt_mu_n_sigma["mu"]) \
                    ==len(trgt_mu_n_sigma["sigma"]) \
                    ==len(prior_dists["mu"]) \
                    ==len(prior_dists["log_sigma"]) \
                    ==len(post_dists["mu"]) \
                    ==len(post_dists["log_sigma"]) \
                    ==len(trgt_ys), \
                "inconsistent training example number"
        
        log_p_list, kl_list = [], []

        for t in range(len_seq):
            dist = trgt_dist_list[t]
            trgt_y_t = trgt_ys[t]
            prior_mu, prior_log_sigma = prior_dists["mu"][t], prior_dists["log_sigma"][t]
            post_mu, post_log_sigma = post_dists["mu"][t], post_dists["log_sigma"][t]
            trgt_mu = trgt_mu_n_sigma["mu"][t]
            
            # 2.2.1 ELBO/log_prob | TrgtNLL/NLL
            log_p = -torch.mean(dist.log_prob(trgt_y_t))
            log_p_list.append(log_p)
            
            # 2.2.2 ELBO/kl
            kl = -0.5*(-prior_log_sigma+post_log_sigma \
                        - (torch.exp(post_log_sigma)+(post_mu-prior_mu)**2) \
                        / torch.exp(prior_log_sigma) + 1.0)
            kl = torch.mean(kl, dim=-1, keepdims=True)
            kl_list.append(torch.mean(kl))
        
        log_p = np.sum(log_p_list) / len(log_p_list)
        kl = np.sum(kl_list) / len(kl_list)
        
        # final loss
        loss = log_p + self._beta * kl

        return loss

    def get_debug_metrics(self, trgt_dist_list, trgt_mu_n_sigma, prior_dists, post_dists, len_seq, trgt_ys, num_cxt_pnts):
        assert trgt_ys, "no training label" 
        assert len_seq==len(trgt_dist_list) \
                    ==len(trgt_mu_n_sigma["mu"]) \
                    ==len(trgt_mu_n_sigma["sigma"]) \
                    ==len(prior_dists["mu"]) \
                    ==len(prior_dists["log_sigma"]) \
                    ==len(post_dists["mu"]) \
                    ==len(post_dists["log_sigma"]) \
                    ==len(trgt_ys), \
                "inconsistent training example number"
        
        log_p_list = []
        log_p_seen, log_p_unseen = [], []
        log_p_wo_cxt, log_p_w_cxt = 0, 0
        mse_list, mse_wo_cxt, mse_w_cxt = [], 0, 0
        cxt_wo, cxt_w = torch.Tensor([0.0]).to(torch.device("cuda")), \
                        torch.Tensor([0.0]).to(torch.device("cuda"))

        for t in range(len_seq):
            dist = trgt_dist_list[t]
            trgt_y_t = trgt_ys[t]
            prior_mu, prior_log_sigma = prior_dists["mu"][t], prior_dists["log_sigma"][t]
            post_mu, post_log_sigma = post_dists["mu"][t], post_dists["log_sigma"][t]
            trgt_mu = trgt_mu_n_sigma["mu"][t]
            
            # 1. ELBO/log_prob | TrgtNLL/NLL
            log_p = -torch.mean(dist.log_prob(trgt_y_t))
            log_p_list.append(log_p)

            '''            
            # 2. TrgtNLL/NLLWithoutCxt
            log_p_wo_cxt += log_p if num_cxt_pnts[t]==0 else torch.Tensor([0.0]).to(torch.device("cuda"))
            cxt_wo += torch.Tensor([1.0]).to(torch.device("cuda")) if num_cxt_pnts[t]==0 else torch.Tensor([0.0]).to(torch.device("cuda"))
            
            # 3. TrgtNLL/NLLSeen
            log_p_seen.append(-1*torch.index_select(log_p, 1, self._index[num_cxt_pnts[t]]))
            # 4. TrgtNLL/NLLUnseen
            log_p_unseen.append(-1*torch.index_select(log_p, 1, self._index[num_cxt_pnts[t]:log_p.shape[1]]))

            # 5. TrgtNLL/NLLWithCxt
            log_p_w_cxt += torch.Tensor([0.0]).to(torch.device("cuda")) if num_cxt_pnts[t]==0 else log_p
            cxt_w += torch.Tensor([0.0]).to(torch.device("cuda")) if num_cxt_pnts[t]==0 else torch.Tensor([1.0]).to(torch.device("cuda"))
            
            # 6. TrgtMSE/MSE
            mse = torch.mean( nn.MSELoss(trgt_y_t, trgt_mu) )
            mse_list.append(mse)
            # 7. TrgtMSE/MSEWithCxt
            mse_w_cxt += torch.Tensor([0.0]).to(torch.device("cuda")) if num_cxt_pnts[t]==0 else mse
            # 8. TrgtMSE/MSEWithoutCxt
            mse_wo_cxt += mse if num_cxt_pnts[t]==0 else torch.Tensor([0.0]).to(torch.device("cuda"))
            '''
        # result merging
        log_p = np.sum(log_p_list) / len(log_p_list)
        '''
        log_p_seen = torch.mean(torch.cat(log_p_seen, dim=-1))
        log_p_unseen = torch.mean(torch.cat(log_p_unseen, dim=-1))
        log_p_w_cxt = torch.Tensor([0.0]).to(torch.device("cuda")) \
                if torch.equal(cxt_w, torch.Tensor([0]).to(torch.device("cuda"))) \
                else log_p_w_cxt / cxt_w
        log_p_wo_cxt = torch.Tensor([0.0]).to(torch.device("cuda")) \
                if torch.equal(cxt_wo, torch.Tensor([0]).to(torch.device("cuda"))) \
                else log_p_wo_cxt / cxt_wo
        mse = np.sum(mse_list) / len(mse_list)
        mse_w_cxt = torch.Tensor([0.0]).to(torch.device("cuda")) \
                if torch.equal(cxt_w, torch.Tensor([0]).to(torch.device("cuda"))) \
                else mse_w_cxt / cxt_w
        mse_wo_cxt = torch.Tensor([0.0]).to(torch.device("cuda")) \
                if torch.equal(cxt_wo, torch.Tensor([0]).to(torch.device("cuda"))) \
                else mse_wo_cxt / cxt_wo
        
        debug_metrics = (log_p, log_p_list, log_p_w_cxt, log_p_wo_cxt,
                        mse, mse_list, mse_w_cxt, mse_wo_cxt, log_p_seen,
                        log_p_unseen)
        '''
        debug_metrics = (log_p)
        return debug_metrics
