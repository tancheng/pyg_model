import torch
import time
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset

from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='/tmp/Cora', name='Cora')

print("dataset[0]: ", dataset[0])

print(dataset)
print("num_classes: ", dataset.num_classes)

import torch.nn.functional as F
from torch_geometric.nn import GCNConv

times = 0

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
#        global times
#        print("times: ", times)
#        times += 1

        x = F.log_softmax(x, dim=1)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model = Net().to(device)
model = Net()
#data = dataset[0].to(device)
data = dataset[0]
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

print("data.y: ", data.y, "; len(data.y): ", len(data.y))
total_used_for_train = 0
for tm in data.train_mask:
  if tm == True:
    total_used_for_train += 1
print("data.train_mask: ", data.train_mask, "; total used for train: ", total_used_for_train)
print("data.y[data.train_mask]: ", data.y[data.train_mask])

model.train()
for epoch in range(200):
    begin = time.time()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    end = time.time()
#    print('time: ', end - begin)


#torch.save(model.state_dict(), "gcn_cora.pt")

#PATH = "./gcn_cora.pt"
#model.load_state_dict(torch.load(PATH))
begin = time.time()
model.eval()
end = time.time()
print('time: ', end - begin)
_, pred = model(data).max(dim=1)
correct = float (pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
acc = correct / data.test_mask.sum().item()
print('Accuracy: {:.4f}'.format(acc))
