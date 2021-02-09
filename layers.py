import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.conv import MessagePassing
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot, zeros


def activation_layer(act_name):
    """Construct activation layers
    Args:
        act_name: str or nn.Module, name of activation function
        hidden_size: int, used for Dice activation
        dice_dim: int, used for Dice activation
    Return:
        act_layer: activation layer
    """
    if isinstance(act_name, str):
        if act_name.lower() == 'sigmoid':
            act_layer = nn.Sigmoid()
        elif act_name.lower() == 'relu':
            act_layer = nn.ReLU()
        elif act_name.lower() == 'leaky_relu':
            act_layer = nn.LeakyReLU(0.1)
    elif issubclass(act_name, nn.Module):
        act_layer = act_name()
    else:
        raise NotImplementedError

    return act_layer


class ResBlock(nn.Module):
    def __init__(self, in_features, use_bn):
        super(ResBlock, self).__init__()
        self.lin1 = nn.Linear(in_features, in_features)
        self.activation = nn.LeakyReLU(0.1)
        self.lin2 = nn.Linear(in_features, in_features)
        self.use_bn = use_bn

        if self.use_bn:
            self.bn1 = nn.BatchNorm1d(in_features)
            self.bn2 = nn.BatchNorm1d(in_features)

    def forward(self, x):
        residual = x

        out = self.lin1(x)
        if self.use_bn:
            out = self.bn1(out)
        out = self.activation(out)

        out = self.lin2(out)
        if self.use_bn:
            out = self.bn2(out)

        out += residual
        out = self.activation(out)

        return out


class ResDNN(nn.Module):
    """The Multi Layer Percetron
      Input shape
        - nD tensor with shape: ``(batch_size, ..., input_dim)``. The most common situation would be a 2D input with shape ``(batch_size, input_dim)``.
      Output shape
        - nD tensor with shape: ``(batch_size, ..., hidden_size[-1])``. For instance, for a 2D input with shape ``(batch_size, input_dim)``, the output would have shape ``(batch_size, hidden_size[-1])``.
      Arguments
        - **inputs_dim**: input feature dimension.
        - **hidden_units**:list of positive integer, the layer number and units in each layer.
        - **activation**: Activation function to use.
        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix.
        - **dropout_rate**: float in [0,1). Fraction of the units to dropout.
        - **use_bn**: bool. Whether use BatchNormalization before activation or not.
        - **seed**: A Python integer to use as random seed.
    """

    def __init__(self, inputs_dim, l_hidden, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False,
                 init_std=0.0001):
        super(ResDNN, self).__init__()
        self.l_hidden = l_hidden
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.l2_reg = l2_reg
        self.use_bn = use_bn

        self.linears = nn.ModuleList(
            [ResBlock(inputs_dim, use_bn=self.use_bn) for i in range(l_hidden)])

        # for name, tensor in self.linears.named_parameters():
        #     if 'weight' in name:
        #         nn.init.normal_(tensor, mean=0, std=init_std)
        # self.to(device)

    def forward(self, x):

        for i in range(self.l_hidden):

            fc = self.linears[i](x)

            fc = self.dropout(fc)
        return fc


class DNN(nn.Module):
    """The Multi Layer Percetron
      Input shape
        - nD tensor with shape: ``(batch_size, ..., input_dim)``. The most common situation would be a 2D input with shape ``(batch_size, input_dim)``.
      Output shape
        - nD tensor with shape: ``(batch_size, ..., hidden_size[-1])``. For instance, for a 2D input with shape ``(batch_size, input_dim)``, the output would have shape ``(batch_size, hidden_size[-1])``.
      Arguments
        - **input_dim**: input feature dimension.
        - **hidden_units**:list of positive integer, the layer number and units in each layer.
        - **activation**: Activation function to use.
        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix.
        - **dropout_rate**: float in [0,1). Fraction of the units to dropout.
        - **use_bn**: bool. Whether use BatchNormalization before activation or not.
        - **seed**: A Python integer to use as random seed.
    """

    def __init__(self, input_dim, hidden_units, activation='leaky_relu', l2_reg=0, dropout_rate=0, use_bn=False,
                 init_std=0.0001):
        super(DNN, self).__init__()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        if len(hidden_units) == 0:
            raise ValueError("hidden_units is empty!!")
        hidden_units = [input_dim] + list(hidden_units)

        self.linears = nn.ModuleList(
            [nn.Linear(hidden_units[i], hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])

        if self.use_bn:
            self.bn = nn.ModuleList(
                [nn.BatchNorm1d(hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])

        self.activation_layers = nn.ModuleList(
            [activation_layer(activation) for i in range(len(hidden_units) - 1)])

        # for name, tensor in self.linears.named_parameters():
        #     if 'weight' in name:
        #         nn.init.normal_(tensor, mean=0, std=init_std)
        # self.to(device)

    def forward(self, x):
        deep_input = x

        for i in range(len(self.linears)):

            fc = self.linears[i](deep_input)

            if self.use_bn:
                fc = self.bn[i](fc)

            fc = self.activation_layers[i](fc)

            fc = self.dropout(fc)
            deep_input = fc
        return deep_input


class CrossNet(nn.Module):
    """The Cross Network part of Deep&Cross Network model,
    which leans both low and high degree cross feature.
      Input shape
        - 2D tensor with shape: ``(batch_size, units)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, units)``.
      Arguments
        - **in_features** : Positive integer, dimensionality of input features.
        - **input_feature_num**: Positive integer, shape(Input tensor)[-1]
        - **layer_num**: Positive integer, the cross layer number
        - **parameterization**: string, ``"vector"``  or ``"matrix"`` ,  way to parameterize the cross network.
        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix
        - **seed**: A Python integer to use as random seed.
      References
        - [Wang R, Fu B, Fu G, et al. Deep & cross network for ad click predictions[C]//Proceedings of the ADKDD'17. ACM, 2017: 12.](https://arxiv.org/abs/1708.05123)
        - [Wang R, Shivanna R, Cheng D Z, et al. DCN-M: Improved Deep & Cross Network for Feature Cross Learning in Web-scale Learning to Rank Systems[J]. 2020.](https://arxiv.org/abs/2008.13535)
    """

    def __init__(self, in_features, layer_num=2, parameterization='matrix'):
        super(CrossNet, self).__init__()
        self.layer_num = layer_num
        self.parameterization = parameterization
        if self.parameterization == 'vector':
            # weight in DCN.  (in_features, 1)
            self.kernels = torch.nn.ParameterList(
                [nn.Parameter(nn.init.xavier_normal_(torch.empty(in_features, 1))) for i in range(self.layer_num)])
        elif self.parameterization == 'matrix':
            # weight matrix in DCN-M.  (in_features, in_features)
            self.kernels = torch.nn.ParameterList([nn.Parameter(nn.init.xavier_normal_(
                torch.empty(in_features, in_features))) for i in range(self.layer_num)])
        else:  # error
            raise ValueError("parameterization should be 'vector' or 'matrix'")

        self.bias = torch.nn.ParameterList(
            [nn.Parameter(nn.init.zeros_(torch.empty(in_features, 1))) for i in range(self.layer_num)])

    def forward(self, inputs):
        x_0 = inputs.unsqueeze(2)
        x_l = x_0
        for i in range(self.layer_num):
            if self.parameterization == 'vector':
                xl_w = torch.tensordot(x_l, self.kernels[i], dims=([1], [0]))
                dot_ = torch.matmul(x_0, xl_w)
                x_l = dot_ + self.bias[i]
            elif self.parameterization == 'matrix':
                dot_ = torch.matmul(self.kernels[i], x_l)  # W * xi  (bs, in_features, 1)
                dot_ = dot_ + self.bias[i]  # W * xi + b
                dot_ = x_0 * dot_  # x0 · (W * xi + b)  Hadamard-product
            else:  # error
                print("parameterization should be 'vector' or 'matrix'")
                pass
            x_l = dot_ + x_l
        x_l = torch.squeeze(x_l, dim=2)
        return x_l


class DiscreteFeatureEmbeddingLayer(nn.Module):
    """DiscreteFeatureEmbeddingLayer"""
    def __init__(self):
        super(DiscreteFeatureEmbeddingLayer, self).__init__()
        self.discrete_emb1 = nn.Linear(3, 2)  # gender
        self.discrete_emb2 = nn.Linear(2, 2)  # age
        self.discrete_emb3 = nn.Linear(7, 2)  # city

    def forward(self, x):
        x1 = self.discrete_emb1(x[:, 0:3])
        x2 = self.discrete_emb2(x[:, 3:5])
        x3 = self.discrete_emb3(x[:, 5:12])
        return torch.cat([x1, x2, x3, x[:, 12:]], dim=1)


class ContinousFeatureEmbeddingLayer(nn.Module):
    """ContinousFeatureEmbeddingLayer"""
    def __init__(self, n_continous_features, out_channels, use_bn=True):
        super(ContinousFeatureEmbeddingLayer, self).__init__()
        self.ncf = n_continous_features
        self.embed1 = DNN(n_continous_features, [out_channels]*2, use_bn=use_bn)
        self.embed2 = DNN(n_continous_features, [out_channels]*2, use_bn=use_bn)
        self.embed3 = DNN(n_continous_features, [out_channels]*2, use_bn=use_bn)

    def forward(self, x):
        x1 = self.embed1(x[:, 0:self.ncf])
        x2 = self.embed2(x[:, 1*self.ncf:2*self.ncf])
        x3 = self.embed3(x[:, 2*self.ncf:3*self.ncf])
        return torch.cat([x1, x2, x3], dim=-1)


class EdgeFeatureEmbeddingLayer(nn.Module):
    """EdgeFeatureEmbeddingLayer"""
    def __init__(self, n_edge_features, out_channels, use_bn=True):
        super(EdgeFeatureEmbeddingLayer, self).__init__()
        self.nef = n_edge_features
        self.embed = DNN(n_edge_features, [n_edge_features]*2, use_bn=use_bn)

    def forward(self, x):
        return self.embed


class PredictionLayer(nn.Module):
    """
      Arguments
         - **task**: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
         - **use_bias**: bool.Whether add bias term or not.
    """

    def __init__(self, input_dim, hidden_units, activation, dropout=0):
        super(PredictionLayer, self).__init__()
        self.input_dim = input_dim
        self.dropout = dropout
        self.act = activation_layer(activation)

        hidden_units = [input_dim] + list(hidden_units) + [1]

        self.linears = nn.ModuleList(
            [nn.Linear(hidden_units[i], hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])

    def forward(self, x):
        output = x
        for i in range(len(self.linears)):
            output = self.linears[i](output)
            if i != len(self.linears) - 1:
                output = self.act(output)
            else:
                output = F.dropout(output, p=self.dropout, training=self.training)
                output = torch.sigmoid(output)
        return output


class EdgeLearning(torch.nn.Module):
    r"""
    learn mask vector of each edge
    Args:
        dim_node: node feature dimensions
        dim_edge: edge feature dimensions

    Output:
        edge mask
    """

    def __init__(self, dim_node, dim_edge, out_dim, activation, **kwargs):
        super(EdgeLearning, self).__init__()
        self.dim_node = dim_node
        self.dim_edge = dim_edge
        self.out_dim = out_dim
        self.act = activation_layer(activation)

        self.fc_i = torch.nn.Linear(dim_node, out_dim)
        self.fc_j = torch.nn.Linear(dim_node, out_dim)
        self.fc_e = torch.nn.Linear(dim_edge, out_dim)
        self.fc = torch.nn.Linear(out_dim, out_dim)

    def forward(self, x, edge_index, edge_attr):
        x_i = x[edge_index[0, :]]
        x_j = x[edge_index[1, :]]
        x_i = self.act(self.fc_i(x_i))
        x_j = self.act(self.fc_j(x_j))
        mask = self.act(self.fc_e(edge_attr))

        mask = x_i + x_j + mask
        mask = self.act(mask)
        mask = self.act(self.fc(mask))

        return mask

    def __repr__(self):
        return '{}(dim_node={}, dim_edge={}, out_dim={})'.format(self.__class__.__name__,
                                                                 self.dim_node,
                                                                 self.dim_edge,
                                                                 self.out_dim)


class ELConv(MessagePassing):
    r"""
    edge learning conv
    Args:
        in_channels: input node feature dimensions
        out_channels: output node feature dimensions
        edge_dim: edge feature dimensions

    """

    def __init__(self, in_channels, out_channels, edge_dim,
                 bias=None, **kwargs):
        super(ELConv, self).__init__(aggr='mean', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim

        self.weight = Parameter(torch.Tensor(
            self.in_channels, self.out_channels))
        self.edge_weight = Parameter(
            torch.Tensor(self.edge_dim, self.out_channels))
        self.bias = Parameter(torch.Tensor(self.out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.edge_weight)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):
        x = torch.matmul(x, self.weight)
        edge_attr = torch.matmul(edge_attr, self.edge_weight)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, edge_attr, x_j):
        return x_j * edge_attr

    def update(self, aggr_out, x):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias

        return aggr_out + x

    def __repr__(self):
        return '{}(in_channels={}, out_channels={})'.format(self.__class__.__name__,
                                                            self.in_channels,
                                                            self.out_channels)
