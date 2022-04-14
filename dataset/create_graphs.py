import pandas as pd
import dgl
import torch
import numpy as np

def create_ids_for_addresses(source, destination):
    unique_addresses = list(set(source+destination))

    address_id_dict = dict()

    for i,address in enumerate(unique_addresses):
        address_id_dict[address] = i

    for i,source_address in enumerate(source):
        source[i] = address_id_dict[source_address]

    for i,destination_address in enumerate(destination):
        destination[i] = address_id_dict[destination_address]
    return source, destination

def create_train_test_val_masks(lst, train_percentage, test_percentage, validate_percentage):
    assert train_percentage+validate_percentage+test_percentage==100
    num_train = int(len(lst)*train_percentage/100)
    num_val = int(len(lst)*validate_percentage/100)
    train_nid = list(range(0,num_train))
    val_nid = list(range(num_train, num_train+num_val)) 
    test_nid = list(range(num_train+num_val, len(lst))) 
    train_mask = torch.zeros((len(lst),), dtype=torch.bool)
    train_mask[train_nid] = True
    val_mask = torch.zeros((len(lst),), dtype=torch.bool)
    val_mask[val_nid] = True
    test_mask = torch.zeros((len(lst),), dtype=torch.bool)
    test_mask[test_nid] = True
    return train_mask, test_mask, val_mask


def main():
    train_percentage = 80
    val_percentage = 10
    test_percentage = 10

    df = pd.read_csv('transaction_data_1M.csv')
    source = df.from_address.tolist()
    destination = df.to_address.tolist()
    #print(source[:10])
    #print(destination[:10])
    #print(len(source), len(destination))

    source, destination = create_ids_for_addresses(source, destination)
    #print(source[:10])
    #print(destination[:10])
    #print(len(source), len(destination))
    n_list = list(set(source+destination))
    train_mask, test_mask, val_mask = create_train_test_val_masks(n_list,train_percentage=train_percentage, test_percentage=test_percentage, validate_percentage=val_percentage)


    g = dgl.graph((source,destination))

    g.ndata['train_mask'] = train_mask
    g.ndata['val_mask'] = val_mask
    g.ndata['test_mask'] = test_mask
    
    # ti = df.transaction_index.to_numpy()
    # val = df.value.to_numpy()
    # print(len(ti),len(val))

    g.edata['transaction_index'] = torch.from_numpy(df.transaction_index.to_numpy())
    g.edata['value'] = torch.from_numpy(df.value.to_numpy(dtype=float))

    # print(df.transaction_index)
    print(f"num_nodes={g.num_nodes()}")
    print(f"num_edges={g.num_edges()}")
    dgl.save_graphs('graph_1M.dgl', [g])

if __name__ == "__main__":
    main()