import pandas as pd
import dgl
import torch
import numpy as np

def main():
    num_parts=4
    (lg,), uk = dgl.load_graphs('../dataset/graph_10M.dgl')
    node_map, edge_map = dgl.distributed.partition_graph(lg,"partitioned_graph_10M",num_parts=num_parts,out_path=f"../../workspace_{num_parts}/partitioned_graphs_10M/",part_method='random',return_mapping=True, balance_ntypes=lg.ndata['train_mask'],balance_edges=True)
if __name__ == "__main__":
    main()