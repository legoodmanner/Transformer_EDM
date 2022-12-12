
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import distributed as dist_fn

class RCBlock(nn.Module):
    def __init__(self, feat_dim, ks, dilation, num_groups):
        super().__init__()
        ksm1 = ks-1
        mfd = feat_dim
        di = dilation
        self.num_groups = num_groups

        self.relu = nn.LeakyReLU()

        # self.rec = nn.GRU(mfd, mfd, num_layers=1, batch_first=True, bidirectional=True)
        self.conv = nn.Conv1d(mfd, mfd, ks, 1, ksm1*di//2, dilation=di, groups=num_groups)
        self.gn = nn.GroupNorm(num_groups, mfd)

    def init_hidden(self, batch_size, hidden_size):
        num_layers = 1
        num_directions = 2
        hidden = torch.zeros(num_layers*num_directions, batch_size, hidden_size)
        hidden.normal_(0, 1)
        return hidden

    def forward(self, x):
        bs, mfd, nf = x.size()
        r = x.clone()
        # hidden = self.init_hidden(bs, mfd).to(x.device)

        # r = x.transpose(1, 2)
        # r, _ = self.rec(r, hidden)
        # r = r.transpose(1, 2).view(bs, 2, mfd, nf).sum(1)
        c = self.relu(self.gn(self.conv(r)))
        x = x+r+c

        return x


class BodyGBlock(nn.Module):
    def __init__(self, input_dim, output_dim, middle_dim, num_groups):
        super().__init__()

        ks = 3  # kernel size
        mfd = middle_dim

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mfd = mfd
        self.num_groups = num_groups

        # ### Main body ###
        block = [
            nn.Conv1d(input_dim, mfd, 3, 1, 1),
            nn.GroupNorm(num_groups, mfd),
            nn.LeakyReLU(),
            RCBlock(mfd, ks, dilation=1, num_groups=num_groups),
            nn.Conv1d(mfd, output_dim, 3, 1, 1),
        ]
        self.block = nn.Sequential(*block)

    def forward(self, x):

        # ### Main ###
        x = self.block(x)

        return x


class NetG(nn.Module):
    def __init__(self, feat_dim, z_dim, z_scale_factors):
        super().__init__()

        mfd = 512
        num_groups = 4
        self.num_groups = num_groups
        self.mfd = mfd

        self.feat_dim = feat_dim
        self.z_dim = z_dim
        self.z_scale_factors = z_scale_factors

        # ### Main body ###
        self.block0 = BodyGBlock(z_dim, mfd, mfd, num_groups)
        self.head0 = nn.Conv1d(mfd, feat_dim, 3, 1, 1)

        blocks = []
        heads = []
        for scale_factor in z_scale_factors:
            block = BodyGBlock(mfd, mfd, mfd, num_groups)
            blocks.append(block)

            head = nn.Conv1d(mfd, feat_dim, 3, 1, 1)
            heads.append(head)

        self.blocks = nn.ModuleList(blocks)
        self.heads = nn.ModuleList(heads)

    def forward(self, z):

        # SBlock0
        z_scale_factors = self.z_scale_factors
        x_body = self.block0(z)
        x_head = self.head0(x_body)

        for ii, (block, head, scale_factor) in enumerate(zip(self.blocks, self.heads, z_scale_factors)):
            x_body = F.interpolate(x_body, scale_factor=scale_factor, mode='nearest')
            x_head = F.interpolate(x_head, scale_factor=scale_factor, mode='nearest')

            x_body = x_body + block(x_body)

            x_head = x_head + head(x_body)

        return x_head


class BNSNConv2dDBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, frequency_stride, time_dilation):
        super().__init__()
        ks = kernel_size
        ksm1d2 = (ks - 1) // 2

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.time_dilation = time_dilation
        self.frequency_stride = frequency_stride

        block = [
            spectral_norm(nn.Conv2d(
                input_dim, output_dim, ks,
                (frequency_stride, 1),
                (1, time_dilation*ksm1d2),
                dilation=(1, time_dilation))),
            nn.BatchNorm2d(output_dim),
            nn.LeakyReLU(),
        ]
        self.block = nn.Sequential(*block)

    def forward(self, x):

        x = self.block(x)

        return x


class BNSNConv1dDBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, dilation):
        super().__init__()
        ks = kernel_size
        ksm1d2 = (ks - 1) // 2

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.dilation = dilation

        block = [
            spectral_norm(nn.Conv1d(input_dim, output_dim, ks,
                                    1, dilation*ksm1d2, dilation=dilation)),
            nn.BatchNorm1d(output_dim),
            nn.LeakyReLU(),
        ]
        self.block = nn.Sequential(*block)

    def forward(self, x):

        x = self.block(x)

        return x


class StridedBNSNConv1dDBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride):
        super().__init__()
        ks = kernel_size
        ksm1d2 = (ks - 1) // 2

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.stride = stride

        block = [
            spectral_norm(nn.Conv1d(input_dim, output_dim, ks, stride, ksm1d2)),
            nn.BatchNorm1d(output_dim),
            nn.LeakyReLU(),
        ]
        self.block = nn.Sequential(*block)

    def forward(self, x):

        x = self.block(x)

        return x


class NetD(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        ks = 3  # kernel size
        mfd = 512

        self.mfd = mfd

        self.input_size = input_size

        # ### Main body ###
        blocks2d = [
            BNSNConv2dDBlock(1, 4, ks, 2, 2),
            BNSNConv2dDBlock(4, 16, ks, 2, 4),
            BNSNConv2dDBlock(16, 64, ks, 2, 8),
        ]

        blocks1d = [
            BNSNConv1dDBlock(64*10, mfd, 3, 1),
            BNSNConv1dDBlock(mfd, mfd, ks, 16),
            BNSNConv1dDBlock(mfd, mfd, ks, 32),
            BNSNConv1dDBlock(mfd, mfd, ks, 64),
            BNSNConv1dDBlock(mfd, mfd, ks, 128),
        ]

        self.body2d = nn.Sequential(*blocks2d)
        self.body1d = nn.Sequential(*blocks1d)

        self.head = spectral_norm(nn.Conv1d(mfd, input_size, 3, 1, 1))

    def forward(self, x):
        '''
        x.shape=(batch_size, feat_dim, num_frames)
        cond.shape=(batch_size, cond_dim, num_frames)
        '''
        bs, fd, nf = x.size()

        # ### Process generated ###
        # shape=(bs, 1, fd, nf)
        x = x.unsqueeze(1)

        # shape=(bs, 64, 10, nf_)
        x = self.body2d(x)
        # shape=(bs, 64*10, nf_)
        x = x.view(bs, -1, x.size(3))

        # ### Merging ###
        x = self.body1d(x)

        # ### Head ###
        # shape=(bs, input_size, nf)
        # out = torch.sigmoid(self.head(x))
        out = self.head(x)

        # Pad
        # out = F.pad(out, pad=(0, nf-out.size(2)))

        return out

class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)#code book vector (64, 512)
        self.register_buffer("embed", embed)#embed in used
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())#embed in updating moving average

    def forward(self, input, bypass=False):
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)#index to one-hot
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0) #[n_embed]
            embed_sum = flatten.transpose(0, 1) @ embed_onehot #[dim, n_embed] sum of the encoder vector output

            dist_fn.all_reduce(embed_onehot_sum)
            dist_fn.all_reduce(embed_sum)

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            ) # exponential moving average of cluster size
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay) # expotentail moving average of embedding.
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            ) #smoothing to prevent clustersize be zero
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        if bypass:
            quantize = input
            # diff = diff.detach()
        else:
            quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))

class VQVAE(nn.Module):
    def __init__(
        self,
        feat_dim,
        z_dim,
        encoder_scale_factors,
        decoder_scale_factors,
        embed_dim,
        n_embed
    ):

    # args:
    #   feat_dim: N of feature shape (N x T), could be frequency/mel bins
    #   z_dim: how many latents to atttend the generation process
    #   encoder/decoder_scale_factor: upsample for decoder, while downsmaple for encoder
    # 
        super().__init__()
        self.encoder = NetG(z_dim, feat_dim, encoder_scale_factors)
        self.decoder = NetG(feat_dim, z_dim, decoder_scale_factors)
        self.quantize_conv = nn.Conv1d(z_dim, embed_dim, 1)
        self.quantize = Quantize(embed_dim, n_embed)
    
    def forward(self, x, bypass=False):
        x = self.encoder(x)
        quant, diff, _id = self.quantizer(x, bypass)
        dec = self.decoder(quant)
        return dec, diff, _id
    
    def quantizer(self, x, bypass=False):
        quant = self.quantize_conv(x).permute(0, 2, 1)
        quant, diff, id = self.quantize(quant, bypass)
        quant = quant.permute(0, 2, 1) #[bs, 20, embed_dim] --> [bs, embed_dim, 20]
        diff = diff.unsqueeze(0)

        return quant, diff, id
    