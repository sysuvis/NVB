import types

import torch
import torch.nn as nn
import torch.nn.functional as F
import typing_extensions


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 2)
        self.out = nn.Linear(2, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.sigmoid(self.out(x))
        return x

# m = MLP()
# print('fc1' in m)
# exit()
# m.fc1 = nn.Linear(2, 4)
# m.fc1.activation = nn.ReLU()
# print(list(m.named_modules()))
# exit()
n = 512
batch_size = 64
x_train1 = (torch.rand([n, 2]) - 0.5) * 2
y_train1 = torch.zeros([n, 1])
for i in range(n):
    if x_train1[i, 0] < x_train1[i, 1] :
        y_train1[i, 0] = 1

x_train = (torch.rand([n, 2]) - 0.5) * 2
y_train = torch.zeros([n, 1])
for i in range(n):
    if x_train[i, 0] ** 2 + x_train[i, 1] ** 2 < 0.3:
        y_train[i, 0] = 1

model = MLP()
init_state_dict = model.state_dict()
from NNVisBuilder import Builder
from NNVisBuilder.Views.Views import *
from NNVisBuilder.Data import *
from NNVisBuilder.Views.Widgets import *


builder = Builder(model, input_=x_train, targets=y_train.squeeze().int().numpy())
# print(builder.name2module['fc1'].activation)
# print(model.fc2.activation)
# exit()
grid = builder.generate_grid_data()
builder.add_hidden_grid(['fc1', 'fc2', 'out'])
builder.add_connects(['fc1', 'fc2', 'out'])

loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
print(optimizer.param_groups[0]['lr'])

record = []
epoch = 4001
for i in range(epoch):
    sample = torch.randint(0, n, size=[batch_size, ])
    # feed the data into model
    pred = model(x_train[sample])

    loss = loss_fn(pred, y_train[sample])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    record.append(loss.data.item())
    if i % 500 == 499:
        print(f'{i+1}/{epoch}')
    if i % 500 == 0:
        builder.record_connect()
        builder.forward_grid_data()
dx = 250
bx = 350
by = 150
dy = 150
s = 100

grid_size = grid.size()[0]
layers = ['fc1', 'fc2', 'out']
out_features = [4, 2, 1]
in_features = [2, 4, 2]
filter = Filter(dim=0, filter_type=Type.Scalar)
embeddings = []
for i in range(len(layers)):
    layer = layers[i]
    embeddings.append(Data(value=builder.embeddings_g[layer]))
    e_data = embeddings[-1].apply_transform(Reshape(shape=[-1, grid_size, *embeddings[-1].size()[1:]]))
    for j in range(out_features[i]):
        view_data = e_data.apply_transform(filter).filter(dim=1, value=j).data()
        view = ScatterPlot(grid, position=[i*dx+bx, j*dy+by], size=[s, s], color_labels=view_data, opacity=0.3)

def modify_weight(i, idx, w):
    print(i)
    print(builder.name2module[layers[i]].weight.size())
    builder.name2module[layers[i]].weight[idx//in_features[i], idx%in_features[i]] = w
    connections[i].update(builder.name2module[layers[i]])

def get_modify_weight(k, link_data):
    def modify_weight(idx, w):
        print(k)
        link_data[idx] = w
        with torch.no_grad():
            builder.name2module[layers[k]].weight[idx // in_features[k], idx % in_features[k]] = w
        # connections[k].update(builder.connections)
    return modify_weight
connections = []
for k in range(len(layers)):
    connections.append(Data(value=builder.connections[layers[k]]))
    link_labels = connections[-1].apply_transform(filter).reshape(-1)
    t1 = []
    t2 = []
    for i in range(out_features[k]):
        for j in range(in_features[k]):
            t1.append(j * dy + by + s / 2)
            t2.append(i * dy + by + s / 2)
    link_nodes = Data(value=[t1, t2])
    link = LinkView(position=[(k-1)*dx+bx+s, 0], size=[dx-s, 0], node_positions=link_nodes, labels=link_labels, colors=link_labels.sign(), width='labels')
    link.onclick(get_modify_weight(k, link_labels))


time_steps = e_data.size()[0]
slider = Slider(position=[100, 120], range=time_steps)
def f(value):
    filter.update(value)
slider.onclick(f)


input_x = Data(value=x_train)
input_y = Data(value=y_train)
input_x1 = Data(value=x_train1)
input_y1 = Data(value=y_train1)
# print(np.max(input_y), np.min(input_y))
view1 = ScatterPlot(input_x, position=[30, 150], size=[50, 50], point_size=1.2, color_labels=input_y, highlight_border=True)
view2 = ScatterPlot(input_x1, position=[30, 230], size=[50, 50], point_size=1.2, color_labels=input_y1, highlight_border=True)

input_x0 = Data(value=x_train)
input_y0 = Data(value=y_train)
view0 = ScatterPlot(input_x0, position=view.align(), point_size=2, color_labels=input_y0)


ScatterPlot(grid, position=[100, 150], size=[s, s], color_labels=grid[:, 0], opacity=0.3)
ScatterPlot(grid, position=[100, 300], size=[s, s], color_labels=grid[:, 1], opacity=0.3)

lr_input = Input(position=[420, 120], size=[30, 20], text=optimizer.param_groups[0]['lr'])
def set_lr(value):
    optimizer.param_groups[0]['lr'] = value
    print(optimizer.param_groups[0]['lr'])
lr_input.onclick(lambda value: set_lr(value))

def f(value):
    # model.load_state_dict(init_state_dict)
    print('d1')
    builder.reset_embedding()
    for i in range(epoch):
        sample = torch.randint(0, n, size=[batch_size, ])
        # feed the data into model
        pred = model(x_train[sample])
        loss = loss_fn(pred, y_train[sample])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        record.append(loss.data.item())
        if i % 500 == 499:
            print(f'{i + 1}/{epoch}')
        if i % 500 == 0:
            builder.record_connect()
            builder.forward_grid_data()
    for i in range(len(layers)):
        layer = layers[i]
        embeddings[i].update(builder.embeddings_g[layer])
        connections[i].update(builder.connections[layer])
    input_x0.update(x_train)
    input_y0.update(y_train)
view1.onclick(f)

def f(value):
    # model.load_state_dict(init_state_dict)
    builder.reset_embedding()
    for i in range(epoch):
        sample = torch.randint(0, n, size=[batch_size, ])
        # feed the data into model
        pred = model(x_train1[sample])
        loss = loss_fn(pred, y_train1[sample])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        record.append(loss.data.item())
        if i % 500 == 499:
            print(f'{i + 1}/{epoch}')
        if i % 500 == 0:
            builder.record_connect()
            builder.forward_grid_data()
    for i in range(len(layers)):
        layer = layers[i]
        embeddings[i].update(builder.embeddings_g[layer])
        connections[i].update(builder.connections[layer])
    input_x0.update(x_train1)

    input_y0.update(y_train1)
view2.onclick(f)


# # construct views
# # for layer in layers:
# #     filter =
# #
# #
# # def f(j):
# #     return lambda data: torch.tensor([x[j] for x in data])
#
#
# # for i in range(0, 2):
# #     for j in range(n[i]):
# #         t.composite_view([
# #             {
# #                 'data': {
# #                     'hidden': l[i],
# #                     'point_size': 2,
# #                     'opacity': 0.1
# #                 },
# #                 'transform': [{
# #                     'activation': torch.tanh,
# #                     'in': 'hidden'
# #                 },{
# #                     'in': 'hidden',
# #                     'out': 'hidden_1',
# #                     'method': f(j)
# #                 }, {
# #                     'in': 'grid',
# #                     'out': 'x',
# #                     'method': lambda data: torch.tensor([x[0] for x in data])
# #                 }, {
# #                     'in': 'grid',
# #                     'out': 'y',
# #                     'method': lambda data: torch.tensor([x[1] for x in data])
# #                 }],
# #                 'element': {
# #                     'class': 'decision_border',
# #                     'color': 'hidden_1',
# #                     'opacity': 'opacity',
# #                     'point_size': 'point_size'
# #                 }
# #             }
# #         ], position=[x[i], 100 + j * 200], size=[160, 160])
# #         # break


builder.run(batch_size=16, foward_manual=True)
