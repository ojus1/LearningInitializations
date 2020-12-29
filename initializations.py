import torch
from torch.distributions import Normal, Uniform
from torch import nn
import math
from torch.nn import init


activation_ids = {
    nn.ReLU: 0,
    nn.Sigmoid: 1,
    nn.Tanh: 2,
    nn.Identity: 3,
    nn.Softmax: 4,
    nn.LogSoftmax: 5
}


class Distribution(nn.Module):
    def __init__(self, kind='uniform'):
        assert(kind in ['uniform', 'normal'])
        super().__init__()

        self.num_features = 2 * \
            self.get_transforms(1).shape[0] + len(activation_ids)

        self.parameter_net = nn.Sequential(
            nn.Linear(self.num_features, self.num_features // 2),
            nn.LeakyReLU(),
            nn.Linear(self.num_features // 2, self.num_features // 4),
            nn.LeakyReLU(),
            nn.Linear(self.num_features // 4, self.num_features // 8),
            nn.LeakyReLU(),
            nn.Linear(self.num_features // 8, 2),
        )
        Distribution.init_params(self.parameter_net)
        self.kind = kind

    def get_weights(self, layer_dict, shape):
        dist = self.get_distribution(
            layer_dict['fan_in'], layer_dict['fan_out'], layer_dict['activation'])
        weights = dist.sample(shape)  # .clamp_(min=-1e3, max=1e3)
        log_prob = dist.log_prob(weights.reshape(-1))
        # print(weights, weights.shape, log_prob, log_prob.shape)
        return weights, log_prob  # / log_prob.numel()

    def get_distribution(self, fan_in, fan_out, activation_id):
        feat = self.create_features(fan_in, fan_out, activation_id)
        params = self.parameter_net(feat).squeeze(0)
        # print(params)
        if self.kind == 'uniform':
            distribution = Uniform(
                params[0], params[1])
        elif self.kind == 'normal':
            distribution = Normal(params[0], params[1])
        # print(params[0], params[1])
        return distribution

    def create_features(self, fan_in, fan_out, activation_id):
        activation_one_hot = self.one_hot(
            activation_ids[activation_id], len(activation_ids))
        fan_in_features = self.get_transforms(fan_in)
        fan_out_features = self.get_transforms(fan_out)

        return torch.cat([
            activation_one_hot,
            fan_in_features,
            fan_out_features
        ],
            dim=0
        ).unsqueeze(0)

    @staticmethod
    def init_params(net):
        '''Init layer parameters.'''
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=1e-3)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    @staticmethod
    def get_transforms(x):
        transforms = [x/1000, x ** 0.5, x ** -0.5, 1/x, math.log(x/100)]
        # for item in transforms:
        #     if math.isnan(item) or math.isinf(item):
        #         raise ValueError()
        #     # print(item)
        return torch.tensor(transforms)

    @staticmethod
    def one_hot(x, size):
        zero = torch.zeros(size)
        zero[x] = 1
        return zero

# @torch.no_grad()


def initialize(target_model, distribution_model):
    log_probs = []
    last_module = None
    for i, m in enumerate(target_model.children()):
        which_activation = {isinstance(
            m, activation): activation for activation in activation_ids.keys()}
        if any(list(which_activation.values())):
            try:
                # print(last_module.weight.shape)
                # print(i, m, last_module)
                layer_dict = {
                    'fan_in': last_module.weight.shape[1],
                    'fan_out': last_module.weight.shape[0],
                    'activation': which_activation[True]
                }
                # print(layer_dict)
                m.weight, log_prob = distribution_model.get_weights(
                    layer_dict, last_module.weight.shape)
                # print(m.weight)
                if torch.isnan(m.weight).any():
                    print("Weight NAN")
                    # exit()
                log_probs.append(log_prob)

                # m.bias, log_prob = distribution_model.get_weights(layer_dict, last_module.bias.shape)
                # log_probs.append(log_prob.unsqueeze(0))
            except Exception as e:
                # import traceback
                # traceback.print_exc()
                # print(e)
                pass
        last_module = m
    return log_probs


def train_step(dist_model, reward, log_probs, optim):
    # with torch.autograd.detect_anomaly():
    optim.zero_grad()

    r_centered = reward / len(log_probs)
    loss = 0
    for i in range(len(log_probs)):
        # print(log_probs[i].shape, torch.isfinite(log_probs[i]).shape)
        finite = log_probs[i][torch.isfinite(log_probs[i])]
        # assert(finite.shape[0] != log_probs[i].shape[0])
        assert(torch.isfinite(finite).all())
        assert(torch.isfinite(finite.sum()).all())
        assert(math.isfinite(r_centered))
        loss += -1 * r_centered * finite.sum()
        if math.isnan(loss.item()):
            print("Loss NAN")
            # exit()

    loss.backward()

    optim.step()
    return loss.item()


if __name__ == "__main__":
    dist_model = Distribution(kind='normal')

    # init_params(dist_model)
    target_model = nn.Sequential(
        nn.Linear(768, 128),
        nn.LeakyReLU(True),
        nn.Linear(128, 10),
        nn.LogSoftmax()
    )

    with torch.no_grad():
        while True:
            log_probs = initialize(target_model, dist_model)
            # print(log_probs)
            if not log_probs[-1].view(-1)[-1] != log_probs[-1].view(-1)[-1]:
                break
            Distribution.init_params(dist_model)

    for i in range(1000):
        reward = -13.2
        reward_mean = -17.2
        reward_std = 4.2

        optim = torch.optim.Adam(dist_model.parameters(), lr=1e-3)
        log_probs = initialize(target_model, dist_model)
        # print(log_probs)
        loss = train_step(dist_model, reward, log_probs, optim)

        # print(loss)
        # break
