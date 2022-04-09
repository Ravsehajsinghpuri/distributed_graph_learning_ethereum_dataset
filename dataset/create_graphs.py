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


def main():
    
    df = pd.read_csv('transaction_data.csv')
    source = df.from_address.tolist()
    destination = df.to_address.tolist()
    print(source[:10])
    print(destination[:10])
    print(len(source), len(destination))

    source, destination = create_ids_for_addresses(source, destination)
    print(source[:10])
    print(destination[:10])
    print(len(source), len(destination))

    g = dgl.graph((source,destination))
    # ti = df.transaction_index.to_numpy()
    # val = df.value.to_numpy()
    # print(len(ti),len(val))

    g.edata['transaction_index'] = torch.from_numpy(df.transaction_index.to_numpy())
    g.edata['value'] = torch.from_numpy(df.value.to_numpy(dtype=float))

    # print(df.transaction_index)
    print(f"num_nodes={g.num_nodes()}")
    print(f"num_edges={g.num_edges()}")
    dgl.save_graphs('graph.dgl', [g])

    (lg,), uk = dgl.load_graphs('graph.dgl')
    print(lg)
    print(uk)

if __name__ == "__main__":
    main()