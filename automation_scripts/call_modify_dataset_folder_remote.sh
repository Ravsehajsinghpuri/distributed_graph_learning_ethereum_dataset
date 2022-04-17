num_servers=1
for i in $( seq 1 $num__servers )
do 
    ssh diml_2022@big_instance_$i 'bash -s' < ./modify_dataset_folder_remote.sh
done