import pandas as pd
import dgl
import torch
import numpy as np



def main():
    
    (lg,), uk = dgl.load_graphs('../dataset/graph.dgl')
    distributed_g = dgl.distributed.partition_graph(lg,"graph_ten",num_parts=4,out_path="output/",part_method='metis')
if __name__ == "__main__":
    main()