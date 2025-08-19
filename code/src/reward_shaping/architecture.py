import torch
import torch.nn as nn

# Initialise using kaiman as its good for relu
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.fill_(0.0)

class RewardNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout_rate=0.3):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.ln_input = nn.LayerNorm(hidden_dim)
        self.block1_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        ])
        self.block1_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim),
            nn.LayerNorm(hidden_dim)
        ])
        self.block2_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        ])
        self.block2_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim),
            nn.LayerNorm(hidden_dim)
        ])
        self.dropout = nn.Dropout(dropout_rate)
        
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.apply(init_weights)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.output_layer.weight, gain=0.01)
            if self.output_layer.bias is not None:
                self.output_layer.bias.fill_(0.0)

    def _residual_block(self, x, layers, norms):
        residual = x
        h = layers[0](x)
        h = norms[0](h)
        h = torch.relu(h)
        h = self.dropout(h)
        h = layers[1](h)
        h = norms[1](h)
        return h + residual

    def forward(self, x):
        h = self.input_projection(x)
        h = self.ln_input(h)
        h = torch.relu(h)
        h = self._residual_block(h, self.block1_layers, self.block1_norms)
        h = torch.relu(h)
        h = self._residual_block(h, self.block2_layers, self.block2_norms)
        h = torch.relu(h)
        return self.output_layer(h)

class RewardNet_No_Res(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout_rate=0.3):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.ln_input = nn.LayerNorm(hidden_dim)
        
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.apply(init_weights)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.output_layer.weight, gain=0.01)
            if self.output_layer.bias is not None:
                self.output_layer.bias.fill_(0.0)

    def forward(self, x):
        h = self.input_projection(x)
        h = self.ln_input(h)
        h = torch.relu(h)
        return self.output_layer(h)