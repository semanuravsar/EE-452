import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class EEGGAT(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=1, heads=(4, 4), dropout=0.2, use_concat=True):
        super().__init__()
        self.use_concat = use_concat

        self.gat1 = GATConv(in_dim, hidden_dim, heads=heads[0], concat=use_concat, dropout=dropout)
        gat1_out_dim = hidden_dim * heads[0] 

        self.gat2 = GATConv(gat1_out_dim, hidden_dim, heads=heads[1], concat=use_concat, dropout=dropout)
        gat2_out_dim = hidden_dim * heads[1] 

        self.head = torch.nn.Linear(gat2_out_dim, hidden_dim)
        self.classifier = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, batch):
        x = self.gat1(x, edge_index)
        x = F.elu(x)

        x = self.gat2(x, edge_index)
        x = F.elu(x)

        x = global_mean_pool(x, batch)

        x = F.relu(self.head(x))
        x = F.dropout(x, p=0.3, training=self.training)

        out = self.classifier(x)      # Graph-level output
        return out.squeeze(1)           # Shape: (batch_size,)
    



class EEGGAT_superpool(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=1, heads=(4, 4), dropout=0.2, use_concat=True):
        super().__init__()
        self.use_concat = use_concat

        self.gat1 = GATConv(in_dim, hidden_dim, heads=heads[0], concat=use_concat, dropout=dropout)
        gat1_out_dim = hidden_dim * heads[0] 

        self.gat2 = GATConv(gat1_out_dim, hidden_dim, heads=heads[1], concat=use_concat, dropout=dropout)
        gat2_out_dim = hidden_dim * heads[1] 

        self.head = torch.nn.Linear(gat2_out_dim, hidden_dim)
        self.classifier = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, batch):
        x = self.gat1(x, edge_index)
        x = F.elu(x)

        x = self.gat2(x, edge_index)
        x = F.elu(x)

        # Get supernode from each graph (assumed at index 19)
        num_graphs = batch.max().item() + 1
        supernode_indices = (torch.arange(num_graphs, device=x.device) * 20) + 19
        x_super = x[supernode_indices]

        x_super = F.relu(self.head(x_super))
        x_super = F.dropout(x_super, p=0.3, training=self.training)

        out = self.classifier(x_super)
        
        return out.squeeze(1)           # Shape: (batch_size,)