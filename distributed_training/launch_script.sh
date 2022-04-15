python3 ~/dgl-master/tools/launch.py \
--workspace ~/workspace2 \
--num_trainers 1 \
--num_samplers 1 \
--num_servers 1 \
--part_config ~/workspace2/partitioned_graphs_1M/partitioned_graph_1M.json \
--ip_config ~/workspace2/ip_config.txt \
"python3 training_scripts/training.py --graph-name 'partitioned_graph_1M' --ip_config ~/workspace2/ip_config.txt"