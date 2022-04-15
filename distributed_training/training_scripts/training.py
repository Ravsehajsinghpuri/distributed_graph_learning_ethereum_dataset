import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
import torch.optim as optim
import numpy as np
from sklearn import metrics
torch.set_default_dtype(torch.float64)
class SAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_layers):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean', activation=torch.nn.ReLU()))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean', activation=torch.nn.ReLU()))
        self.layers.append(dglnn.SAGEConv(n_hidden, 1, 'mean', activation=None))

    def forward(self, blocks, x):
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            x = layer(block, x)
            if l != self.n_layers - 1:
                x = F.relu(x)
        return x

def logger(string):
    with open("logs.txt","a+") as logged_file:
        logged_file.write(string)

def main():
    dgl.distributed.initialize(ip_config='ip_config.txt')
    torch.distributed.init_process_group(backend='gloo')

    g = dgl.distributed.DistGraph('partitioned_graph_1M')

    train_nid = dgl.distributed.node_split(g.ndata['train_mask'])
    valid_nid = dgl.distributed.node_split(g.ndata['val_mask'])

    num_hidden = 256
    num_layers = 2
    lr = 0.001
    model = SAGE(g.ndata['feat'].shape[1], num_hidden, num_layers)
    model = torch.nn.parallel.DistributedDataParallel(model)
    loss_fcn = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    sampler = dgl.dataloading.MultiLayerNeighborSampler([25,10])
    train_dataloader = dgl.dataloading.DistNodeDataLoader(
                                g, train_nid, sampler, batch_size=1,
                                shuffle=True, drop_last=False)
    valid_dataloader = dgl.dataloading.DistNodeDataLoader(
                                g, valid_nid, sampler, batch_size=1,
                                shuffle=False, drop_last=False)

    for epoch in range(10):
        # Loop over the dataloader to sample mini-batches.
        losses = []
        with model.join():
            for step, (input_nodes, seeds, blocks) in enumerate(train_dataloader):
                # Load the input features as well as output labels
                batch_inputs = g.ndata['feat'][input_nodes]
                batch_labels = g.ndata['labels'][seeds]
                # Compute loss and prediction
                batch_pred = model(blocks, batch_inputs)
                loss = loss_fcn(torch.squeeze(batch_pred), batch_labels)
                
                optimizer.zero_grad()
                loss.backward()
                logger('Epoch {}: Training Error {}\n'.format(epoch, loss))
                losses.append(loss.detach().cpu().numpy())
                optimizer.step()

        # validation
        predictions = []
        labels = []
        with torch.no_grad(), model.join():
            for step, (input_nodes, seeds, blocks) in enumerate(valid_dataloader):
                inputs = g.ndata['feat'][input_nodes]
                labels.append(g.ndata['labels'][seeds].numpy())
                predictions.append(model(blocks, inputs).numpy())
            predictions = np.concatenate(predictions)
            labels = np.concatenate(labels)
            mae = metrics.mean_absolute_error(labels, predictions)
            logger('Epoch {}: Validation Error {}'.format(epoch, mae))

if __name__ == "__main__":
    main()