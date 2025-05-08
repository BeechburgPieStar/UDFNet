import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax

def conv3x3(in_planes, out_planes, stride=1, bias=False):
    "3x3 convolution without padding"
    return ModuleParallel(nn.Conv2d(in_planes, out_planes, kernel_size=3,
                                    stride=stride, padding=0, bias=bias))

def conv3x3_p(in_planes, out_planes, stride=1, bias=False):
    "3x3 convolution with padding"
    return ModuleParallel(nn.Conv2d(in_planes, out_planes, kernel_size=3,stride=stride, padding=1, bias=bias))

def conv1x1(in_planes, out_planes, stride=1, bias=False):
    "1x1 convolution"
    return ModuleParallel(nn.Conv2d(in_planes, out_planes, kernel_size=1,
                                    stride=stride, padding=0, bias=bias))

class Exchange(nn.Module):
    def __init__(self):
        super(Exchange, self).__init__()

    def forward(self, x, bn, bn_threshold):
        bn1, bn2 = bn[0].weight.abs(), bn[1].weight.abs()
        x1, x2 = torch.zeros_like(x[0]), torch.zeros_like(x[1])
        x1[:, bn1 >= bn_threshold] = x[0][:, bn1 >= bn_threshold]
        x1[:, bn1 < bn_threshold] = x[1][:, bn1 < bn_threshold]
        x2[:, bn2 >= bn_threshold] = x[1][:, bn2 >= bn_threshold]
        x2[:, bn2 < bn_threshold] = x[0][:, bn2 < bn_threshold]
        return [x1, x2]

class ModuleParallel(nn.Module):
    def __init__(self, module):
        super(ModuleParallel, self).__init__()
        self.module = module

    def forward(self, x_parallel):
        return [self.module(x) for x in x_parallel]

class BatchNorm2dParallel(nn.Module):
    def __init__(self, num_features, num_parallel):
        super(BatchNorm2dParallel, self).__init__()
        for i in range(num_parallel):
            setattr(self, 'bn_' + str(i), nn.BatchNorm2d(num_features))

    def forward(self, x_parallel):
        return [getattr(self, 'bn_' + str(i))(x) for i, x in enumerate(x_parallel)]

class Bottleneck(nn.Module):

    def __init__(self, planes,expansion, num_parallel, bn_threshold, stride=1):
        super(Bottleneck, self).__init__()
        self.midplane = planes//expansion
        self.conv1 = conv1x1(planes, self.midplane)
        self.bn1 = BatchNorm2dParallel(self.midplane, num_parallel)
        self.conv2 = conv3x3_p(self.midplane, self.midplane, stride=stride)
        self.bn2 = BatchNorm2dParallel(self.midplane, num_parallel)
        self.conv3 = conv1x1(self.midplane, planes)
        self.bn3 = BatchNorm2dParallel(planes, num_parallel)

        self.relu = ModuleParallel(nn.ReLU(inplace=True))
        self.num_parallel = num_parallel
        self.exchange = Exchange()
        self.bn_threshold = bn_threshold
        self.bn2_list = []
        for module in self.bn2.modules():
            if isinstance(module, nn.BatchNorm2d):
                self.bn2_list.append(module)

    def forward(self, x):
        residual = x
        out = x
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # if len(x) > 0:
        out = self.exchange(out, self.bn2_list, self.bn_threshold)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = [out[l] + residual[l] for l in range(self.num_parallel)]
        out = self.relu(out)

        return out

class Dropout(nn.Module):
    def __init__(self):
        super(Dropout, self).__init__()
    def forward(self, x):
        out = F.dropout(x, p=0.2, training=self.training)
        return out


class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SCConv(nn.Module):
    def __init__(self, inplanes, planes, pooling_r =2):
        super(SCConv, self).__init__()

        self.k1 = nn.Sequential(nn.Conv2d(planes, planes, kernel_size=3,  padding =1, bias=False),
                    nn.BatchNorm2d(planes),)
        self.k2 = nn.Sequential(nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),
                    nn.Conv2d(planes, planes, kernel_size=3, padding = 1, bias=False),
                    nn.BatchNorm2d(planes), )
        self.k3 = nn.Sequential(nn.Conv2d(planes, planes, kernel_size=3, padding = 1, bias=False),
                    nn.BatchNorm2d(planes),)
        self.k4 = nn.Sequential(nn.Conv2d(planes, planes, kernel_size=3,  padding = 1, bias=False),
                    nn.BatchNorm2d(planes),)
        self.conv1_a = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1_a = nn.BatchNorm2d(planes)
        self.conv1_b = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1_b = nn.BatchNorm2d(planes)

    def forward(self, x):

        out_a = self.conv1_a(x)
        identity = out_a
        out_a = self.bn1_a(out_a)
        out_b = self.conv1_b(x)
        out_b = self.bn1_b(out_b)
        out_a = F.relu(out_a)
        out_b = F.relu(out_b)

        out = torch.sigmoid(torch.add(identity, F.interpolate(self.k2(out_a), identity.size()[2:])))
        out = torch.mul(self.k3(out_a), out)
        out1 = self.k4(out)
        out2 = self.k1(out_b)
        out = torch.cat((out1,out2),1)
        return out

class External_attention(nn.Module):

    def __init__(self, c):
        super(External_attention, self).__init__()
        
        self.conv1 = nn.Conv2d(c, c, 1)
        self.k = c//4
        self.linear_0 = nn.Conv1d(c, self.k, 1, bias=False)
        self.linear_1 = nn.Conv1d(self.k, c, 1, bias=False)
        # self.linear_1.weight = self.linear_0.weight.permute(1, 0, 2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c))        

        self.relu = nn.ReLU()

    def forward(self, x):
        idn = x
        x = self.conv1(x)
        b, c, h, w = x.size()
        x = x.view(b, c, h*w)
        attn = self.linear_0(x)
        attn = F.softmax(attn, dim=-1)
        attn = attn / (1e-9 + attn.sum(dim=1, keepdims=True))
        x = self.linear_1(attn)
        x = x.view(b, c, h, w)
        x = self.conv2(x)
        x = x + idn
        x = self.relu(x)
        return x

class Classifier(nn.Module):
    def __init__(self,  hidden_size, num_classes):
        super(Classifier, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(hidden_size*2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        out = self.fc2(F.relu(self.fc1(x)))
        return out


def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)


class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""

    def __init__(self, in_dim):
        super(CrissCrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2,
                                                                                                                 1)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2,
                                                                                                                 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H) + self.INF(m_batchsize, height, width)).view(m_batchsize, width,
                                                                                                     height,
                                                                                                     height).permute(0,
                                                                                                                     2,
                                                                                                                     1,
                                                                                                                     3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        # print(concate)
        # print(att_H)
        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)
        # print(out_H.size(),out_W.size())
        return self.gamma * (out_H + out_W) + x


class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""
    def __init__(self, in_dim):
        super(CrissCrossAttention,self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))


    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
        #print(concate)
        #print(att_H)
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
        #print(out_H.size(),out_W.size())
        return self.gamma*(out_H + out_W) + x


class Netqmf(nn.Module):
    def __init__(self, hsi_channels, sar_channels, hidden_size, block, num_parallel, num_reslayer=2, num_classes=7,
                 bn_threshold=2e-2):
        self.planes = hidden_size
        self.num_parallel = num_parallel
        self.expansion = 2

        super(Netqmf, self).__init__()

        self.conv_00 = nn.Sequential(nn.Conv2d(hsi_channels, hidden_size, 1, bias=False),
                                     nn.BatchNorm2d(hidden_size))
        self.conv_11 = nn.Sequential(nn.Conv2d(sar_channels, hidden_size, 1, bias=False),
                                     nn.BatchNorm2d(hidden_size))

        self.conv1 = ModuleParallel(nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=0, bias=False))
        self.bn1 = BatchNorm2dParallel(hidden_size, num_parallel)
        self.relu = ModuleParallel(nn.ReLU(inplace=True))
        self.layer = self._make_layer(block, hidden_size, num_reslayer, bn_threshold)

        self.classifier = Classifier(hidden_size, num_classes)
        self.Attention = External_attention(hidden_size)
        self.SCConv = SCConv(hidden_size, hidden_size)
        self.alpha = nn.Parameter(torch.ones(num_parallel, requires_grad=True))
        self.register_parameter('alpha', self.alpha)
        self.ccnet = CrissCrossAttention(128)

    def _make_layer(self, block, planes, num_blocks, bn_threshold, stride=1):
        layers = []
        layers.append(block(planes, self.expansion, self.num_parallel, bn_threshold, stride))
        for i in range(1, num_blocks):
            layers.append(block(planes, planes, self.num_parallel, bn_threshold))
        return nn.Sequential(*layers)

    def forward(self, x, y):
        x = F.relu(self.conv_00(x)).unsqueeze(0)
        y = F.relu(self.conv_11(y)).unsqueeze(0)
        x = torch.cat((x, y), 0)
        x = self.relu(self.bn1(self.conv1(x)))
        out = self.layer(x)

        ens = 0
        alpha_soft = F.softmax(self.alpha, dim=0)
        for l in range(self.num_parallel):
            ens += alpha_soft[l] * out[l].detach()
        out.append(ens)

        hsi = self.ccnet(out[0])
        sar = self.ccnet(out[1])
        hsi_logits = self.classifier(self.SCConv(self.Attention(hsi)))
        sar_logits = self.classifier(self.SCConv(self.Attention(sar)))

        hsi_energy = torch.log(torch.sum(torch.exp(hsi_logits), dim=1))
        sar_energy = torch.log(torch.sum(torch.exp(sar_logits), dim=1))

        hsi_conf0 = hsi_energy / 10
        sar_conf0 = sar_energy / 10
        hsi_conf = torch.reshape(hsi_conf0, (-1, 1))
        sar_conf = torch.reshape(sar_conf0, (-1, 1))

        hsi_sar_logits = (hsi_logits * hsi_conf.detach() + sar_logits * sar_conf.detach())


        return hsi_sar_logits, hsi_logits, sar_logits, hsi_conf0, sar_conf0


if __name__ == '__main__':
    device = 'cuda:0'
    x = torch.randn(64, 180, 7, 7).to(device)
    y = torch.randn(64, 4, 7, 7).to(device)
    model = Netqmf(hsi_channels=180, sar_channels=4, hidden_size=128, block=Bottleneck, num_parallel=2, num_reslayer=2, num_classes=7,
                 bn_threshold=2e-2).to(device)
    hsi_sar_logits, hsi_logits, sar_logits, hsi_conf0, sar_conf0 = model(x, y)


