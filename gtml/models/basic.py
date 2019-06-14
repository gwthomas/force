import torch
import torch.nn as nn
import torch.nn.functional as F


def infer_shape(module, in_shape):
    with torch.no_grad():
        fake_in = torch.zeros([1] + list(in_shape))
        out = module(fake_in)
        return out.shape[1:]
    
    
class FlattenLayer(nn.Module):
    def __init__(self, size):
        super(FlattenLayer, self).__init__()
        self.size = size
    
    def forward(self, x):
        return x.view(-1, self.size)

    
def mlp(sizes, activation=nn.ReLU):
    cg = []
    for i in range(len(sizes) - 1):
        cg.append(nn.Linear(sizes[i], sizes[i+1]))
        cg.append(activation())
    cg.pop(-1) # remove nonlinearity before final activation
    return nn.Sequential(*cg)


def convnet(in_shape, conv_layer_infos, fc_sizes, activation=nn.ReLU):
    prev_channels = in_shape[0]
    conv_layers = []
    for i, (n_filters, kernel_size) in enumerate(conv_layer_infos):
        conv = nn.Conv2d(prev_channels, n_filters, kernel_size)
        conv_layers.append(conv)
        conv_layers.append(activation())
        prev_channels = n_filters
    conv_module = nn.Sequential(*conv_layers)
    conv_out_shape = infer_shape(conv_module, in_shape)
    conv_flat_size = torch.prod(torch.tensor(conv_out_shape))
    fc_module = mlp([conv_flat_size] + fc_sizes, activation)
    return nn.Sequential(conv_module, FlattenLayer(conv_flat_size), fc_module)