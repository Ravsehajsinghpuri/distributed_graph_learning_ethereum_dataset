num_parts=0
for workspace_id in 1 2 4 8
    mkdir ~/workspace_$workspace_id/partitioned_graphs_1M
    for i in $( seq 0 $num_parts )
    do
        mv ~/workspace_$workspace_id/part$i ~/workspace_$workspace_id/partitioned_graphs_1M/
    done
    mv ~/workspace_$workspace_id/partitioned_graph_1M.json ~/workspace_$workspace_id/partitioned_graphs_1M/