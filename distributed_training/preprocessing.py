import pandas as pd
import dgl
import torch
import numpy as np



def main():
    
    (lg,), uk = dgl.load_graphs('../dataset/graph_1M.dgl')
    node_map, edge_map = dgl.distributed.partition_graph(lg,"partitioned_graph_1M",num_parts=4,out_path="partitioned_graphs_1M/",part_method='random',return_mapping=True, balance_ntypes=lg.ndata['train_mask'],balance_edges=True)
if __name__ == "__main__":
    main()