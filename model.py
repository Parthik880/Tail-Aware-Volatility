import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import yfinance as yf
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class DenseLayer1D(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv1 = nn.Conv1d(in_channels, 4 * growth_rate, kernel_size=1, bias=False)

        self.bn2 = nn.BatchNorm1d(4 * growth_rate)
        self.conv2 = nn.Conv1d(
            4 * growth_rate,
            growth_rate,
            kernel_size=3,
            padding=1,
            bias=False
        )

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return torch.cat([x, out], dim=1)



class DenseBlock1D(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate):
        super().__init__()
        layers = []
        channels = in_channels

        for _ in range(num_layers):
            layers.append(DenseLayer1D(channels, growth_rate))
            channels += growth_rate

        self.block = nn.Sequential(*layers)
        self.out_channels = channels

    def forward(self, x):
        return self.block(x)
class Transition1D(nn.Module):
    def __init__(self, in_channels, compression=0.5):
        super().__init__()
        out_channels = int(in_channels * compression)

        self.bn = nn.BatchNorm1d(in_channels)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv(F.relu(self.bn(x)))
        x = self.pool(x)
        return x
class DenseNet1D(nn.Module):
    def __init__(
        self,
        in_channels=49,        # number of input features
        growth_rate=32,
        block_layers=(6, 12, 24, 16),
        compression=0.5,
        out_dim=1
    ):
        super().__init__()

        # Initial convolution
        self.conv0 = nn.Conv1d(
            in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.bn0 = nn.BatchNorm1d(64)
        self.pool0 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        channels = 64

        # Dense blocks
        self.blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()

        for i, num_layers in enumerate(block_layers):
            block = DenseBlock1D(num_layers, channels, growth_rate)
            channels = block.out_channels
            self.blocks.append(block)

            if i != len(block_layers) - 1:
                trans = Transition1D(channels, compression)
                channels = trans.out_channels
                self.transitions.append(trans)

        self.bn_final = nn.BatchNorm1d(channels)
        self.time_pool = nn.AdaptiveAvgPool1d(15)

# force feature dimension = 512
        self.proj = nn.Conv1d(channels, 512, kernel_size=1)
 
        #self.fc = nn.Linear(channels, out_dim)

    def forward(self, x):
        x=x.permute(0,2,1)
        # x: (B, C, T)
        x = self.pool0(F.relu(self.bn0(self.conv0(x))))

        for i, block in enumerate(self.blocks):
            x = block(x)
            if i < len(self.transitions):
                x = self.transitions[i](x)

        x = F.relu(self.bn_final(x))
        x=self.time_pool(x)
        x=self.proj(x)

        # Global average pooling over time
        # (B, C)

        return x.permute(0,2,1)








class Basic(torch.nn.Module):
    expansion=1
    def __init__(self,in_channels,out_channels,stride=1):
        super().__init__()
        self.conv1=torch.nn.Conv1d(in_channels,out_channels,kernel_size=3,padding=1)
        self.bn1=torch.nn.BatchNorm1d(out_channels)
        self.conv2=torch.nn.Conv1d(out_channels,out_channels,kernel_size=3,padding=1)
        self.relu=torch.nn.ReLU()
        self.bn2=torch.nn.BatchNorm1d(out_channels)
        self.shortcut=torch.nn.Identity()
        if stride!=1 or in_channels!=out_channels:
            self.shortcut=torch.nn.Sequential(
                torch.nn.Conv1d(in_channels,out_channels,kernel_size=1,padding=0),
                torch.nn.BatchNorm1d(out_channels)
            )
      
    def forward(self,x):
        identity=self.shortcut(x)
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)
        out=self.conv2(out)
        out=self.bn2(out)
        out+=identity
        out=self.relu(out)
        return out

def make_layer(block, in_channels, out_channels, num_blocks):
    layers = []

    # First block may downsample
    layers.append(block(in_channels, out_channels))

    # Remaining blocks keep same shape
    for _ in range(1, num_blocks):
        layers.append(block(out_channels, out_channels))

    return torch.nn.Sequential(*layers)
   
    
class Res34(torch.nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.conv1=torch.nn.Conv1d(in_channels=in_channels,out_channels=64,kernel_size=3,padding=1)
        self.maxpool = torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.bn=torch.nn.BatchNorm1d(64)
        self.relu=torch.nn.ReLU()
       # self.max=torch.nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.layer1 = make_layer(Basic, 64,  64,  num_blocks=3)
        self.layer2 = make_layer(Basic, 64,  128, num_blocks=4)
        self.layer3 = make_layer(Basic, 128, 256, num_blocks=6)
        self.layer4 = make_layer(Basic, 256, 512, num_blocks=3)
        #self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        #self.fc = torch.nn.Linear(512, )
    def forward(self,x):
        x=x.permute(0,2,1)
        out=self.conv1(x)
        out=self.bn(out)
        out=self.relu(out)
        out=self.maxpool(out)
#-----------------------------------------
        out=self.layer1(out)
        out=self.layer2(out)
        out=self.layer3(out)
        out=self.layer4(out)
#-----------------------------------------
        #out=self.avgpool(out)
        #out=torch.flatten(out,1)
        #out=self.fc(out)
        out=out.permute(0,2,1)
        return out







#attention model
#-----------------------------------------

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()

        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.qkv = nn.Linear(n_embd, 3*n_embd)
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        weights = torch.softmax(scores, dim=-1)

        out = weights @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.proj(out)
def el(x):
    return F.elu(x)+1

class Multi_Head_Linear_Attention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()

        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.qkv = nn.Linear(n_embd, 3*n_embd)
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        kv=el(k).transpose(-2,-1)@el(v)
        Z=el(q)@el(k).sum(dim=-2).unsqueeze(-1)
        out=(el(q)@kv)/Z
        out=out.transpose(1,2).contiguous().view(B,T,C)
        out = self.proj(out)   
        return out

#Multi Layer Perceptron
class MLP(nn.Module):
    def __init__(self,n_embd):
        super().__init__()
        self.l1=nn.Linear(n_embd,4*n_embd)
        self.gelu=nn.GELU()
        self.l2=nn.Linear(n_embd*4,n_embd)
    def forward(self,x):
        x=self.l2(self.gelu(self.l1(x)))
        return x



#Transformer Block
class Block(nn.Module):#linear attention block
    def __init__(self,n_embd,n_head):
        super().__init__()
        self.att=Multi_Head_Linear_Attention(n_embd,n_head)
        self.mlp=MLP(n_embd)
        self.ln1=nn.RMSNorm(n_embd)
        self.ln2=nn.RMSNorm(n_embd)
    def forward(self,x):
        x=x+self.att(self.ln1(x))
        x=x+self.mlp(self.ln2(x))
        return x
    
class Block1(nn.Module):#softmax attention
    def __init__(self,n_embd,n_head):
        super().__init__()
        self.att=MultiHeadAttention(n_embd,n_head)
        self.mlp=MLP(n_embd)
        self.ln1=nn.RMSNorm(n_embd)
        self.ln2=nn.RMSNorm(n_embd)
    def forward(self,x):
        x=x+self.att(self.ln1(x))
        x=x+self.mlp(self.ln2(x))
        return x

#attention pooling with time decay
class selfattnpooling(nn.Module):
     def __init__(self, n_embd, max_len=512):
        super().__init__()
        self.score = nn.Linear(n_embd, 1)

        # Learnable temperature to smooth attention
        self.temperature = nn.Parameter(torch.tensor(1.0))

        # Time decay bias buffer
        self.register_buffer(
            "time_decay",
            torch.linspace(0, 1, max_len).view(1, max_len, 1)
        )

     def forward(self, x):
        B, T, D = x.shape

        scores = self.score(x)

        # Add recency bias (favor recent timesteps)
        scores = scores + self.time_decay[:, :T] * 2.0

        # Smooth attention
        weights = F.softmax(scores / self.temperature, dim=1)

        pooled = (weights * x).sum(dim=1)
        return pooled
    



class Volatility_Prediction_Model(nn.Module):
    def __init__(self,n_embd,n_head,block_size,n_blocks):
        super().__init__()
        self.block_size=block_size
        #self.wpe=nn.Embedding(15,n_embd)
        self.res=Res34(49)
        self.dense=DenseNet1D(49)
        self.fusion = nn.Linear(2*512, n_embd)
        self.rnn = nn.GRU(n_embd, n_embd, batch_first=True)
        self.blocks=nn.Sequential(*[Block(n_embd,n_head) for _ in range(n_blocks-1)])
        self.b1=Block1(n_embd,n_head)
        self.ln=nn.RMSNorm(n_embd)
        self.l1=nn.Linear(n_embd,1)
        self.pool=selfattnpooling(n_embd)
        #self.scale = nn.Parameter(torch.tensor(1.0))
        self.base_head = nn.Linear(n_embd, 1)   # log baseline vol
        self.tail_head = nn.Linear(n_embd, 1)   # tail intensity


    def forward(self,x):
        #short=self.short_cnn(x)
        #long=self.long_cnn(x)
        #x=torch.cat([short,long],dim=-1)
        long=self.res(x)
        short=self.dense(x)
        x=self.fusion(torch.cat([short,long],dim=-1))
        x, _ = self.rnn(x)
        #pos=self.wpe(torch.arange(0,15,dtype=torch.long, device=x.device))
        #x=x+pos
        x=self.b1(self.blocks(x))
        x=self.ln(x)
        
        x=self.pool(x)
        log_sigma_base = self.base_head(x)          # (B,1)
        tail_amp = F.softplus(self.tail_head(x))    # ≥ 0

        log_sigma = log_sigma_base + torch.log1p(tail_amp)
        return log_sigma
