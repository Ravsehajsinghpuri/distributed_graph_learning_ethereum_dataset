# distributed_graph_learning_ethereum_dataset

USE DGL version 0.7.1 to avoid deadlocking/packet loss issue. Issue exists in v0.8 

1) Run pull_data.py from dataset folder --- fetches data from big query and saves to csv file
2) Run create_graphs.py from dataset folder --- creates dgl graph with required metadata from csv file
3) Run preprocessing.py from distributed_training folder --- partitions the dgl graph for distributed training
4) Copy required files including copy_graph_partitions.sh into correspoding workspace folder
5) Run copy_graph_partitions.sh from workspace/partitioned_graphs_1M folder
6) Run distributed_training/launch_script.sh
