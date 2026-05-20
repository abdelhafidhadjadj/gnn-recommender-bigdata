from .graphsage import GraphSAGE_Recommender
from .gat import GAT_Recommender
from .lightgcn import LightGCN_Recommender


def build_model(model_type: str, num_nodes: int,
                emb_dim: int, dropout: float,
                gat_heads: int = 4,
                n_layers: int = 1,
                use_residual: bool = True):
    """
    Factory for all recommender GNN architectures.

    model_type: "sage" | "gat" | "lightgcn"
    n_layers   : number of graph conv layers
    use_residual: add skip-connection after each layer (SAGE/GAT only)
    """
    if model_type == "sage":
        return GraphSAGE_Recommender(num_nodes, emb_dim, dropout,
                                     n_layers, use_residual)
    if model_type == "gat":
        return GAT_Recommender(num_nodes, emb_dim, gat_heads, dropout,
                               n_layers, use_residual)
    if model_type == "lightgcn":
        return LightGCN_Recommender(num_nodes, emb_dim, n_layers)
    raise ValueError(f"Unknown model_type {model_type!r}. Choose: sage | gat | lightgcn")
