import torch
import torch.nn.functional as F
import torch_geometric
import numpy as np

"""
This GNN was mostly taken from an example at Ecole.ai (https://www.ecole.ai/)
The exact example can be seen here: https://github.com/ds4dm/ecole/blob/master/examples/branching-imitation.ipynb
The design comes from a paper by Gasse et.al Exact Combinatorial Optimisation with Graph Convolutional Neural Networks
See https://arxiv.org/abs/1906.01629 for the paper itself.

The design was intended to generate branching scores over candidate variables, so the output naturally has 
dimension n (number of variables or columns). We bastardise this approach by using the design for cut selection instead
of the intended purpose. The number of cut selector parameters is in no way related to the size of the bipartite graph
representation of the instance. We thus make our final output embedding size equal to the number of desired
cut selector parameters and aggregate our result over the columns post forward pass. 

We originally planeed to base our design on the MIPLIB feature vector
(see https://link.springer.com/article/10.1007/s12532-020-00194-3), but this was insufficient from preliminary results.
Whether this was a flaw in the computation design or a limit of the MIPLIB feature vector is yet to be shown.
"""


class GNNPolicy(torch.nn.Module):
    """
    This is the larger GNN
    """

    def __init__(self):
        super().__init__()
        self.emb_size = 32
        self.n_col_features = 7
        self.n_row_features = 7
        self.n_edge_features = 1
        self.n_cutsel_params = 4

        # Constraint (Row) Embedding
        self.row_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(self.n_row_features),
            torch.nn.Linear(self.n_row_features, self.emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.emb_size, self.emb_size),
            torch.nn.ReLU()
        )

        # Edge Embedding
        self.edge_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(self.n_edge_features)
        )

        # Variable (Column) Embedding
        self.col_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(self.n_col_features),
            torch.nn.Linear(self.n_col_features, self.emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.emb_size, self.emb_size),
            torch.nn.ReLU()
        )

        # Half-convolution from columns to row
        self.conv_col_to_row = BipartiteGraphConvolution()
        # Half-convolution from rows back to columns
        self.conv_row_to_col = BipartiteGraphConvolution()

        # Final layers on the transformed column representation
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(self.emb_size, self.emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.emb_size, self.n_cutsel_params, bias=False),
            torch.nn.ReLU()
        )

    def forward(self, edge_indices, coefficients, col_features, row_features):
        # Reverse the edge indices for the half-convolution in the backwards direction
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        # First step: Embed the features per row and column into a common dimension
        row_features = self.row_embedding(row_features)
        col_features = self.col_embedding(col_features)
        # The edge features (normalised coefficients) do not get embedded in a higher dimension
        edge_features = self.edge_embedding(coefficients)

        # Now do to convolutions, first from cols -> rows and then rows-> cols
        row_features = self.conv_col_to_row(edge_indices, edge_features, col_features, row_features)
        col_features = self.conv_row_to_col(reversed_edge_indices, edge_features, row_features, col_features)

        # The final step is to do a MLP on the column features. The choice of using the transformed column features
        # is somewhat arbitrary as the cut selector parameters aren't necessarily preferentially dependant on the
        # columns of the LP. We just want to see the effectiveness of this now standard design
        output = self.output_module(col_features).squeeze(-1)

        return torch.mean(torch.nn.functional.normalize(output, p=1, dim=1), dim=0)


class BipartiteGraphConvolution(torch_geometric.nn.MessagePassing):
    """
    The bipartite graph convolution is already provided by pytorch geometric and we merely need
    to provide the exact form of the messages being passed.
    For documentation on this see https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html
    """

    def __init__(self):
        super().__init__('add')
        # Add here refers to the aggregation of the neighbourhood
        self.emb_size = 32

        self.feature_module_left = torch.nn.Sequential(
            torch.nn.Linear(self.emb_size, self.emb_size)
        )

        self.feature_module_edge = torch.nn.Sequential(
            torch.nn.Linear(1, self.emb_size, bias=False)
        )

        self.feature_module_right = torch.nn.Sequential(
            torch.nn.Linear(self.emb_size, self.emb_size, bias=False)
        )

        self.feature_module_final = torch.nn.Sequential(
            torch.nn.LayerNorm(self.emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.emb_size, self.emb_size)
        )

        self.post_conv_module = torch.nn.Sequential(
            torch.nn.LayerNorm(self.emb_size)
        )

        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(2 * self.emb_size, self.emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.emb_size, self.emb_size)
        )

    def forward(self, edge_indices, coefficients, left_features, right_features):
        # This performs the transformations over the graph topology by calling message(). It also aggregates results
        output = self.propagate(edge_indices, size=(left_features.shape[0], right_features.shape[0]),
                                node_features=(left_features, right_features), edge_features=coefficients)
        # Concatenate the old features and the new features before throwing them through some additional Linear Layers
        return self.output_module(torch.cat([self.post_conv_module(output), right_features], dim=-1))

    def message(self, node_features_i, node_features_j, edge_features):
        output = self.feature_module_final(self.feature_module_left(node_features_i)
                                           + self.feature_module_edge(edge_features)
                                           + self.feature_module_right(node_features_j))
        return output
