# yes | sudo apt-get update
# yes | sudo apt-get -y install python3-pip
# yes | pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
# yes | pip3 install dgl dglgo -f https://data.dgl.ai/wheels/repo.html
# yes | pip3 install sklearn
# yes | sudo apt-get install nfs-kernel-server
# mkdir -p /home/diml_2022/workspace_1
# mkdir -p /home/diml_2022/workspace_2
# mkdir -p /home/diml_2022/workspace_4
# mkdir -p /home/diml_2022/workspace_8
# echo "/home/diml_2022/workspace_1 10.168.0.0/16(rw,sync,no_subtree_check)" | sudo tee -a /etc/exports
# echo "/home/diml_2022/workspace_2 10.168.0.0/16(rw,sync,no_subtree_check)" | sudo tee -a /etc/exports
# echo "/home/diml_2022/workspace_4 10.168.0.0/16(rw,sync,no_subtree_check)" | sudo tee -a /etc/exports
# echo "/home/diml_2022/workspace_8 10.168.0.0/16(rw,sync,no_subtree_check)" | sudo tee -a /etc/exports



#conda
yes | wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
'' yes '' | bash -b -p Anaconda3-2020.02-Linux-x86_64.sh
yes | conda create --name diml_2022 python=3.9
yes | conda activate diml_2022
yes | conda install pip
yes | pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
yes | pip3 install dgl dglgo -f https://data.dgl.ai/wheels/repo.html
yes | pip3 install sklearn