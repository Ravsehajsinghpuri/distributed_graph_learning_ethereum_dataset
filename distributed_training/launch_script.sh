/home/diml_2022/anaconda3/envs/diml_2022/bin/python ~/dgl-master/tools/launch.py \
--workspace ~/workspace_4 \
--num_trainers 1 \
--num_samplers 1 \
--num_servers 1 \
--num_omp_threads 1 \
--part_config ~/workspace_4/partitioned_graphs_1M/partitioned_graph_1M.json \
--ip_config ~/workspace_4/ip_config.txt \
"/home/diml_2022/anaconda3/envs/diml_2022/bin/python training_scripts/train_dist.py --graph_name 'partitioned_graph_1M' --ip_config ~/workspace_4/ip_config.txt --part_config ~/workspace_4/partitioned_graphs_1M/partitioned_graph_1M.json --batch_size_eval 1000"