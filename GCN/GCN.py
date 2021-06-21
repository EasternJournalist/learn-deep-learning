import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, PairNorm
from torch_geometric.utils.undirected import to_undirected
import random
import matplotlib.pyplot as plt

data_name = 'citeseer'      # 'cora' or 'citeseer'
data_edge_path = f'datasets/{data_name}/{data_name}.cites'
data_content_path = f'datasets/{data_name}/{data_name}.content'

raw_content = pd.read_table(data_content_path, header=None, dtype={0:np.str})
raw_edge = pd.read_table(data_edge_path, header=None, dtype=np.str)

paper_ids = raw_content[0]
paper_id_map = {}
for i, pp_id in enumerate(paper_ids):
    paper_id_map[pp_id] = i

edge_index = torch.from_numpy(raw_edge.apply(lambda col: col.map(paper_id_map)).dropna().values).long().t().contiguous()
x = torch.from_numpy(raw_content.values[:, 1:-1].astype(np.float)).float()

labels = np.unique(raw_content[raw_content.keys()[-1]]).tolist()
y = torch.from_numpy(raw_content[raw_content.keys()[-1]].map(lambda x: labels.index(x)).values).long()

def get_mask(y:torch.tensor):
    train_mask = torch.tensor([False] * y.shape[0])
    for i in torch.unique(y).unbind():
        temp = torch.arange(0, y.shape[0])[y == i].tolist()
        random.shuffle(temp)
        train_mask[temp[:30]] = True
    
    train_mask = torch.tensor(train_mask)
    test_mask = train_mask == False
    return train_mask, test_mask

train_mask, test_mask = get_mask(y)
data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, test_mask=test_mask)

def drop_edge(edge_index, keep_ratio:float=1.):
    num_keep = int(keep_ratio * edge_index.shape[1])
    temp = [True] * num_keep + [False] * (edge_index.shape[1] - num_keep)
    random.shuffle(temp)
    return edge_index[:, temp]

class GCNNodeClassifier(torch.nn.Module):
    def __init__(self, 
        dim_features, 
        num_classes, 
        num_layers, 
        add_self_loops:bool=True, 
        use_pairnorm:bool=False, 
        drop_edge:float=1., 
        activation:str='relu', 
        undirected:bool=False
    ):
        super(GCNNodeClassifier, self).__init__()
        dim_hidden = 32

        self.gconvs = torch.nn.ModuleList(
            [GCNConv(in_channels=dim_features, out_channels=dim_hidden, add_self_loops=add_self_loops)] 
            + [GCNConv(in_channels=dim_hidden, out_channels=dim_hidden, add_self_loops=add_self_loops) for i in range(num_layers - 2)]
        )
        self.final_conv = GCNConv(in_channels=dim_hidden, out_channels=num_classes, add_self_loops=add_self_loops)

        self.use_pairnorm = use_pairnorm
        if self.use_pairnorm:
            self.pairnorm = PairNorm()
        self.drop_edge = drop_edge
        activations_map = {'relu':torch.relu, 'tanh':torch.tanh, 'sigmoid':torch.sigmoid, 'leaky_relu':torch.nn.LeakyReLU(0.1)}
        self.activation_fn = activations_map[activation]
        
    def forward(self, x, edge_index):
        for l in self.gconvs:
            edges = drop_edge(edge_index, self.drop_edge)
            x = l(x, edges)
            if self.use_pairnorm:
                x = self.pairnorm(x)
            x = self.activation_fn(x)
        x = self.final_conv(x, edge_index)

        return x

def eval_acc(y_pred, y):
    return ((torch.argmax(y_pred, dim=-1) == y).float().sum() / y.shape[0]).item()

num_epochs = 100
test_cases = [
    {'num_layers':2, 'add_self_loops':True, 'use_pairnorm':False, 'drop_edge':1., 'activation':'relu', 'undirected':False}, 
    # num layers
    {'num_layers':4, 'add_self_loops':True, 'use_pairnorm':False, 'drop_edge':1., 'activation':'relu', 'undirected':False}, 
    {'num_layers':6, 'add_self_loops':True, 'use_pairnorm':False, 'drop_edge':1., 'activation':'relu', 'undirected':False}, 
    # self loop
    {'num_layers':2, 'add_self_loops':False, 'use_pairnorm':False, 'drop_edge':1., 'activation':'relu', 'undirected':False}, 
    # pair norm
    {'num_layers':2, 'add_self_loops':True, 'use_pairnorm':True, 'drop_edge':1., 'activation':'relu', 'undirected':False}, 
    {'num_layers':4, 'add_self_loops':True, 'use_pairnorm':True, 'drop_edge':1., 'activation':'relu', 'undirected':False}, 
    {'num_layers':6, 'add_self_loops':True, 'use_pairnorm':True, 'drop_edge':1., 'activation':'relu', 'undirected':False}, 
    # drop edge
    {'num_layers':2, 'add_self_loops':True, 'use_pairnorm':False, 'drop_edge':0.6, 'activation':'relu', 'undirected':False}, 
    {'num_layers':4, 'add_self_loops':True, 'use_pairnorm':False, 'drop_edge':0.6, 'activation':'relu', 'undirected':False}, 
    # activation fn
    {'num_layers':2, 'add_self_loops':True, 'use_pairnorm':False, 'drop_edge':1., 'activation':'tanh', 'undirected':False}, 
    {'num_layers':2, 'add_self_loops':True, 'use_pairnorm':False, 'drop_edge':1., 'activation':'leaky_relu', 'undirected':False}, 
    # undirected
    {'num_layers':2, 'add_self_loops':True, 'use_pairnorm':False, 'drop_edge':1., 'activation':'relu', 'undirected':True}, 
    {'num_layers':4, 'add_self_loops':True, 'use_pairnorm':True, 'drop_edge':1., 'activation':'relu', 'undirected':True}, 
    {'num_layers':4, 'add_self_loops':True, 'use_pairnorm':False, 'drop_edge':0.8, 'activation':'relu', 'undirected':True}, 
]

for i_case, kwargs in enumerate(test_cases):
    print(f'Test Case {i_case:>2}')
    model = GCNNodeClassifier(x.shape[1], len(labels), **kwargs)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    history_test_acc = []
    input_edge_index = to_undirected(edge_index) if kwargs['undirected'] else edge_index
    
    for i_epoch in range(0, num_epochs):
        print(f'Epoch {i_epoch:>3} ', end='')
        y_pred = model(x, input_edge_index)
        train_acc = eval_acc(y_pred[train_mask], y[train_mask])

        # Train
        loss = F.cross_entropy(y_pred[train_mask], y[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Test
        test_acc = eval_acc(y_pred[test_mask], y[test_mask])
        history_test_acc.append(test_acc)
        print(f'Train Acc = {train_acc}. Test Acc = {test_acc}')
    kwargs['best_acc'] = max(history_test_acc)
    plt.plot(list(range(num_epochs)), history_test_acc, label=f'case_{str(i_case).zfill(2)}')

plt.legend()
plt.savefig(f'{data_name}-HistoryAcc.jpg')
pd.DataFrame(test_cases).to_csv(f'{data_name}-Result.csv')
    