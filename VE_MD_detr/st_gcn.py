import torch
import torch.nn as nn
import math

class GraphConv(nn.Module):
    """
    Simple Graph Convolution for a batched adjacency matrix A of shape (B*T*M, V, V).
    """
    def __init__(self, in_channels, out_channels, bias=True):
        super(GraphConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Weight: (in_channels, out_channels)
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        # Initialize
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(in_channels)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, A):
        """
        x: (B, T, M, C, V)
        A: (B, T, M, V, V)
        Returns: (B, T, M, out_channels, V)
        """
        B, T, M, C, V = x.shape
        # Flatten (B, T, M) => btm
        btm = B * T * M

        # Reshape x => (btm, C, V) 
        x = x.reshape(btm, C, V)
        # Reshape A => (btm, V, V)
        A = A.reshape(btm, V, V)

        # Graph conv step:  x_out = A @ x @ W
        #   A shape: (btm, V, V)
        #   x shape: (btm, C, V)
        # We'll do multiplication in two steps or using einsum

        # 1) adjacency multiplication along V dimension
        #    result => (btm, C, V)
        x = torch.einsum('bvw,bcw->bcv', A, x)

        # 2) linear transform along C dimension => (C->out_channels)
        #    x shape: (btm, C, V), weight shape: (C, out_channels)
        x = torch.einsum('bcv,co->bov', x, self.weight)

        if self.bias is not None:
            x = x + self.bias.unsqueeze(-1)  # shape (out_channels,) -> broadcast along V

        # x => (btm, out_channels, V) => reshape back to (B, T, M, out_channels, V)
        x = x.reshape(B, T, M, self.out_channels, V)
        return x


class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(STGCNBlock, self).__init__()
        # Temporal conv: convolve over T dimension (kernel_size=3).
        # We'll treat (C, T, V, M) as: channels=in_channels, height=T, width=V*M (just an example).
        # Or more commonly do a Conv2d with (kernel_size=(3,1)).
        self.temporal_conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(3,1,1),  # no convolve over T
            padding=(1,0,0),
            stride=(1,1,1)
        )
        self.graph_conv = GraphConv(in_channels, out_channels)
        
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        """
        x: (B, T, M, C, V)
        A: (B, T, M, V, V) 
        """
        x = x.contiguous()
        B, T, M, C, V = x.shape
        x = self.graph_conv(x, A)    # => (B, T, M, out_channels, V)
 
        # Our conv3d expects (B, C, D, H, W). We'll interpret T as D, V as H, M as W:
        x = x.reshape(B, C, T, V, M)
        x = self.temporal_conv(x) # shape still (B, C, T, V, M), but channels might be in_channels -> in_channels

        x = self.bn(x)  
        x = self.relu(x)
        x = x.reshape(B, T, M, C, V) 
        return x




class STGCN(nn.Module):
    def __init__(self, in_channels=2):
        super(STGCN, self).__init__()
        # Example: two ST-GCN blocks
        self.stgcn = STGCNBlock(in_channels, in_channels)  # 2 -> 2
        
    def forward(self, x, A):
        """
        x: (B, T,M, C, V)
        A: (B, T, M, V, V)
        """
        # Pass through ST-GCN blocks
        x = self.stgcn(x, A)   
        return x














