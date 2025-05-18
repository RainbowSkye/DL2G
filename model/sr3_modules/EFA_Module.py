import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import numpy as np


class ConvNormAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, norm=None,
                 num_groups=8, act='relu', negative_slope=0.1, inplace=False, reflect=False):
        super(ConvNormAct, self).__init__()
        self.layer = nn.Sequential()
        if reflect:
            self.layer.add_module('pad', nn.ReflectionPad2d(padding))
            self.layer.add_module('conv',
                                  nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                            stride=stride, padding=0, dilation=dilation, bias=bias))
        else:
            self.layer.add_module('conv',
                                  nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                            stride=stride, padding=padding, dilation=dilation, bias=bias))
        if norm == 'bn':
            self.layer.add_module('norm', nn.BatchNorm2d(num_features=out_channels))
        elif norm == 'gn':
            self.layer.add_module('norm', nn.GroupNorm(num_groups=num_groups, num_channels=out_channels))
        else:
            pass
        if act == 'relu':
            self.layer.add_module('act', nn.ReLU(inplace=inplace))
        if act == 'relu6':
            self.layer.add_module('act', nn.ReLU6(inplace=inplace))
        elif act == 'elu':
            self.layer.add_module('act', nn.ELU(alpha=1.0))
        elif act == 'lrelu':
            self.layer.add_module('act', nn.LeakyReLU(negative_slope=negative_slope, inplace=inplace))
        elif act == 'sigmoid':
            self.layer.add_module('act', nn.Sigmoid())
        elif act == 'tanh':
            self.layer.add_module('tanh', nn.Tanh())
        else:
            pass

    def forward(self, x):
        y = self.layer(x)
        return y


class Encoder(nn.Module):
    def __init__(self, in_channels=3, temp_channels=512):
        super(Encoder, self).__init__()
        self.encoder_conv1 = nn.Sequential(
            ConvNormAct(in_channels, 64, kernel_size=3, padding=1, norm='bn', act='relu'),
            ConvNormAct(64, 64, kernel_size=3, padding=1, norm='bn', act='relu'),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.encoder_conv2 = nn.Sequential(
            ConvNormAct(64, 128, kernel_size=3, padding=1, norm='bn', act='relu'),
            ConvNormAct(128, 128, kernel_size=3, padding=1, norm='bn', act='relu'),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.encoder_conv3 = nn.Sequential(
            ConvNormAct(128, 256, kernel_size=3, padding=1, norm='bn', act='relu'),
            ConvNormAct(256, 256, kernel_size=3, padding=1, norm='bn', act='relu'),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.encoder_conv4 = nn.Sequential(
            ConvNormAct(256, 512, kernel_size=3, padding=1, norm='bn', act='relu'),
            ConvNormAct(512, 512, kernel_size=3, padding=1, norm=None, act=None)
        )

    def forward(self, x):
        conv1 = self.encoder_conv1(x)
        conv2 = self.encoder_conv2(conv1)
        conv3 = self.encoder_conv3(conv2)
        conv4 = self.encoder_conv4(conv3)
        return conv4


class Decoder(nn.Module):
    def __init__(self, temp_channels=1024, out_channels=3):
        super(Decoder, self).__init__()

        def upsample(in_channels, out_channels):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=False)
            )

        self.decoder_conv1 = nn.Sequential(
            ConvNormAct(temp_channels, 512, kernel_size=3, padding=1, norm='bn', act='relu'),
            ConvNormAct(512, 512, kernel_size=3, padding=1, norm='bn', act='relu')
        )
        self.up1 = upsample(512, 512)
        self.decoder_conv2 = nn.Sequential(
            ConvNormAct(512, 256, kernel_size=3, padding=1, norm='bn', act='relu'),
            ConvNormAct(256, 256, kernel_size=3, padding=1, norm='bn', act='relu')
        )
        self.up2 = upsample(256, 256)
        self.decoder_conv3 = nn.Sequential(
            ConvNormAct(256, 128, kernel_size=3, padding=1, norm='bn', act='relu'),
            ConvNormAct(128, 128, kernel_size=3, padding=1, norm='bn', act='relu')
        )
        self.up3 = upsample(128, 128)
        self.decoder_conv4 = nn.Sequential(
            ConvNormAct(128, 64, kernel_size=3, padding=1, norm='bn', act='relu'),
            ConvNormAct(64, 64, kernel_size=3, padding=1, norm='bn', act='relu')
        )
        self.final_conv = ConvNormAct(64, 3, kernel_size=3, padding=1, norm=None, act='tanh')


    def forward(self, fea):
        conv1 = self.decoder_conv1(fea)
        up1 = self.up1(conv1)

        conv2 = self.decoder_conv2(up1)
        up2 = self.up2(conv2)

        conv3 = self.decoder_conv3(up2)
        up3 = self.up3(conv3)

        conv4 = self.decoder_conv4(up3)
        output = self.final_conv(conv4)

        return output


def random_uniform(shape, low, high, cuda):
    x = torch.rand(*shape)
    result_cpu = (high - low) * x + low
    if cuda:
        return result_cpu.cuda()
    else:
        return result_cpu


def distance(a, b):
    return torch.sqrt(((a - b) ** 2).sum()).unsqueeze(0)


def distance_batch(a, b):
    bs, _ = a.shape
    result = distance(a[0], b)
    for i in range(bs - 1):
        result = torch.cat((result, distance(a[i], b)), 0)

    return result


def multiply(x):  # to flatten matrix into a vector
    return functools.reduce(lambda x, y: x * y, x, 1)


def flatten(x):
    """ Flatten matrix into a vector """
    count = multiply(x.size())
    return x.resize_(count)


def index(batch_size, x):
    idx = torch.arange(0, batch_size).long()
    idx = torch.unsqueeze(idx, -1)
    return torch.cat((idx, x), dim=1)


def MemoryLoss(memory):
    m, d = memory.size()
    memory_t = torch.t(memory)
    similarity = (torch.matmul(memory, memory_t)) / 2 + 1 / 2  # 30X30
    identity_mask = torch.eye(m).cuda()
    sim = torch.abs(similarity - identity_mask)

    return torch.sum(sim) / (m * (m - 1))


class Memory(nn.Module):
    def __init__(self, memory_size, feature_dim, key_dim, temp_update, temp_gather):
        super(Memory, self).__init__()
        # Constants
        self.memory_size = memory_size
        self.feature_dim = feature_dim
        self.key_dim = key_dim
        self.temp_update = temp_update
        self.temp_gather = temp_gather

    def hard_neg_mem(self, mem, i):
        similarity = torch.matmul(mem, torch.t(self.keys_var))
        similarity[:, i] = -1
        _, max_idx = torch.topk(similarity, 1, dim=1)

        return self.keys_var[max_idx]

    def random_pick_memory(self, mem, max_indices):

        m, d = mem.size()
        output = []
        for i in range(m):
            flattened_indices = (max_indices == i).nonzero()
            a, _ = flattened_indices.size()
            if a != 0:
                number = np.random.choice(a, 1)
                output.append(flattened_indices[number, 0])
            else:
                output.append(-1)

        return torch.tensor(output)

    def get_update_query(self, mem, max_indices, update_indices, score, query, train):

        m, d = mem.size()
        if train:
            query_update = torch.zeros((m, d)).cuda()
            random_update = torch.zeros((m, d)).cuda()
            for i in range(m):
                idx = torch.nonzero(max_indices.squeeze(1) == i)
                a, _ = idx.size()
                if a != 0:
                    query_update[i] = torch.sum(((score[idx, i] / torch.max(score[:, i])) * query[idx].squeeze(1)),
                                                dim=0)
                else:
                    query_update[i] = 0

            return query_update

        else:
            query_update = torch.zeros((m, d)).cuda()
            for i in range(m):
                idx = torch.nonzero(max_indices.squeeze(1) == i)
                a, _ = idx.size()
                if a != 0:
                    query_update[i] = torch.sum(((score[idx, i] / torch.max(score[:, i])) * query[idx].squeeze(1)),
                                                dim=0)
                else:
                    query_update[i] = 0

            return query_update

    def get_score(self, mem, query):
        bs, h, w, d = query.size()
        m, d = mem.size()

        score = torch.matmul(query, torch.t(mem))
        score = score.view(bs * h * w, m)

        score_query = F.softmax(score, dim=0)
        score_memory = F.softmax(score, dim=1)

        return score_query, score_memory

    def forward(self, query, keys, train=True):

        batch_size, dims, h, w = query.size()
        query = F.normalize(query, dim=1)
        query = query.permute(0, 2, 3, 1)

        # train
        if train:
            # gathering loss
            gathering_loss = self.gather_loss(query, keys, train)
            # spreading_loss
            spreading_loss = self.spread_loss(query, keys, train)
            # read
            updated_query, softmax_score_query, softmax_score_memory = self.read(query, keys)
            # update
            updated_memory = self.update(query, keys, train)

            return updated_query, updated_memory, softmax_score_query, softmax_score_memory, gathering_loss, spreading_loss

        # test
        else:
            # gathering loss
            gathering_loss = self.gather_loss(query, keys, train)

            # read
            updated_query, softmax_score_query, softmax_score_memory = self.read(query, keys)

            # update
            updated_memory = keys

            return updated_query, updated_memory, softmax_score_query, softmax_score_memory, gathering_loss

    def update(self, query, keys, train):

        batch_size, h, w, dims = query.size()

        softmax_score_query, softmax_score_memory = self.get_score(keys, query)

        query_reshape = query.contiguous().view(batch_size * h * w, dims)

        _, gathering_indices = torch.topk(softmax_score_memory, 1, dim=1)
        _, updating_indices = torch.topk(softmax_score_query, 1, dim=0)

        if train:
            query_update = self.get_update_query(keys, gathering_indices, updating_indices, softmax_score_query,
                                                 query_reshape, train)
            updated_memory = F.normalize(query_update + keys, dim=1)

        else:
            query_update = self.get_update_query(keys, gathering_indices, updating_indices, softmax_score_query,
                                                 query_reshape, train)
            updated_memory = F.normalize(query_update + keys, dim=1)

        return updated_memory.detach()

    def pointwise_gather_loss(self, query_reshape, keys, gathering_indices, train):
        n, dims = query_reshape.size()
        loss_mse = torch.nn.MSELoss(reduction='none')

        pointwise_loss = loss_mse(query_reshape, keys[gathering_indices].squeeze(1).detach())

        return pointwise_loss

    def spread_loss(self, query, keys, train):
        batch_size, h, w, dims = query.size()

        loss = torch.nn.TripletMarginLoss(margin=1.0)

        softmax_score_query, softmax_score_memory = self.get_score(keys, query)

        query_reshape = query.contiguous().view(batch_size * h * w, dims)

        _, gathering_indices = torch.topk(softmax_score_memory, 2, dim=1)

        pos = keys[gathering_indices[:, 0]]
        neg = keys[gathering_indices[:, 1]]

        spreading_loss = loss(query_reshape, pos.detach(), neg.detach())

        return spreading_loss

    def gather_loss(self, query, keys, train):

        batch_size, h, w, dims = query.size()

        loss_mse = torch.nn.MSELoss()

        softmax_score_query, softmax_score_memory = self.get_score(keys, query)

        query_reshape = query.contiguous().view(batch_size * h * w, dims)

        _, gathering_indices = torch.topk(softmax_score_memory, 1, dim=1)

        gathering_loss = loss_mse(query_reshape, keys[gathering_indices].squeeze(1).detach())

        return gathering_loss

    def read(self, query, updated_memory):
        batch_size, h, w, dims = query.size()

        softmax_score_query, softmax_score_memory = self.get_score(updated_memory, query)

        query_reshape = query.contiguous().view(batch_size * h * w, dims)

        concat_memory = torch.matmul(softmax_score_memory.detach(), updated_memory)

        updated_query = torch.cat((query_reshape, concat_memory), dim=1)
        updated_query = updated_query.view(batch_size, h, w, 2 * dims)
        updated_query = updated_query.permute(0, 3, 1, 2)

        return updated_query, softmax_score_query, softmax_score_memory


class EFA(nn.Module):
    def __init__(self, channels=3, dims=512):
        super().__init__()
        self.encoder = Encoder(channels, dims)
        self.memory = Memory(memory_size=dims, feature_dim=dims, key_dim=dims, temp_update=0.1, temp_gather=0.1)
        self.decoder = Decoder(2*dims, channels)

    def forward(self, x, keys, train=True):
        feat = self.encoder(x)
        if train:
            updated_query, keys, softmax_score_query, softmax_score_memory, gathering_loss, spreading_loss = self.memory(
                feat, keys, train)
        else:
            spreading_loss = None
            updated_query, keys, softmax_score_query, softmax_score_memory, gathering_loss = self.memory(feat, keys, train)
        updated_feat = self.decoder(updated_query)
        return updated_feat, keys, gathering_loss, spreading_loss
