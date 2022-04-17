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
    assert train_percentage+test_percentage+validate_percentage==100
    num_train = int(len(lst)*train_percentage/100)
    num_test = int(len(lst)*test_percentage/100)
    train_nid = list(range(0,num_train))
    test_nid = list(range(num_train, num_train+num_test))
    val_nid = list(range(num_train+num_test, len(lst))) 
    train_mask = torch.zeros((len(lst),), dtype=torch.bool)
    train_mask[train_nid] = True
    test_mask = torch.zeros((len(lst),), dtype=torch.bool)
    test_mask[test_nid] = True
    val_mask = torch.zeros((len(lst),), dtype=torch.bool)
    val_mask[val_nid] = True
    return train_mask, test_mask, val_mask

def create_features_and_labels_graph(source, destination, transaction_value):
    unique_addresses = list(set(source+destination))

    labels_balance = []
    node_features = []
    
    incoming_dict = dict()
    outgoing_dict = dict()

    in_count_dict = dict()
    out_count_dict = dict()
    
    for address, value in list(zip(source, transaction_value)):
        bal = outgoing_dict.setdefault(address,0)
        outgoing_dict[address] = bal+value

        out_count = out_count_dict.setdefault(address,0)
        out_count_dict[address] = out_count + 1
    
    for address, value in list(zip(destination, transaction_value)):
        bal = incoming_dict.setdefault(address,0)
        incoming_dict[address] = bal+value
        
        in_count = in_count_dict.setdefault(address,0)
        in_count_dict[address] = in_count + 1
    
    bal_stat = [incoming_dict.setdefault(address,0) - outgoing_dict.setdefault(address,0) for address in unique_addresses]
    print(np.mean(bal_stat))
    for address in unique_addresses:
        feat = [in_count_dict.setdefault(address,0), out_count_dict.setdefault(address,0)]
        node_features.append(feat)

        bal = incoming_dict.setdefault(address,0) - outgoing_dict.setdefault(address,0)
        if bal>0:
            bal_label = 1
        else:
            bal_label = 0
        labels_balance.append(bal_label)
    
    return node_features, labels_balance


def main():
    train_percentage = 70
    test_percentage = 10
    val_percentage = 20

    df = pd.read_csv('transaction_data_1M.csv')
    source = df.from_address.tolist()
    destination = df.to_address.tolist()
    transaction_values = df.value.to_numpy(dtype=np.double)
    #print(source[:10])
    #print(destination[:10])
    #print(len(source), len(destination))

    source, destination = create_ids_for_addresses(source, destination)
    #print(source[:10])
    #print(destination[:10])
    #print(len(source), len(destination))
    n_list = list(set(source+destination))
    train_mask, test_mask, val_mask = create_train_test_val_masks(n_list,train_percentage=train_percentage,test_percentage=test_percentage, validate_percentage=val_percentage)

    node_features, labels = create_features_and_labels_graph(source, destination, transaction_values)

    g = dgl.graph((source,destination))

    g.ndata['train_mask'] = train_mask
    g.ndata['val_mask'] = val_mask
    g.ndata['test_mask'] = test_mask

    g.ndata['features'] = torch.from_numpy(np.array(node_features, dtype=np.double))
    g.ndata['labels'] = torch.from_numpy(np.array(labels, dtype=np.double))
    
 
    # ti = df.transaction_index.to_numpy()
    # val = df.value.to_numpy()
    # print(len(ti),len(val))

    # g.edata['transaction_index'] = torch.from_numpy(df.transaction_index.to_numpy(dtype=np.double))
    # g.edata['value'] = torch.from_numpy(transaction_values)

    # print(g.ndata['features'][:10],g.ndata['labels'][:10],g.edata['transaction_index'][:10],g.edata['value'][:10])
    # print(df.transaction_index)
    print(f"num_nodes={g.num_nodes()}")
    print(f"num_edges={g.num_edges()}")
    dgl.save_graphs('graph_1M.dgl', [g])

if __name__ == "__main__":
    main()