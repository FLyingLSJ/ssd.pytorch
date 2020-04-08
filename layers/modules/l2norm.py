import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import Variable
import torch.nn.init as init


class L2Norm(nn.Module):
    '''
    conv4_3特征图大小38x38，网络层靠前，norm较大，需要加一个L2 Normalization,
    以保证和后面的检测层差异不是很大，具体可以参考： ParseNet。
    '''
    def __init__(self,n_channels, scale):
        super(L2Norm,self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight,self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        #x /= norm
        x = torch.div(x,norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out

        
# 测试代码    
if __name__ == "__main__":
    x = torch.randn(1, 512, 38, 38)
    l2norm = L2Norm(512, 20)
    out = l2norm(x)
    print('L2 norm :', out.shape)
    # L2 norm : torch.Size([1, 512, 38, 38])