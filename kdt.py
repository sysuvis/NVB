import numpy as np
from torch_geometric.datasets import Planetoid
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GATConv
import torch.optim as optim
from NNVisBuilder.Data import *
from NNVisBuilder.Views.Views import *
from NNVisBuilder.Views.Widgets import *
from NNVisBuilder.GlobalVariables import *
from NNVisBuilder import Builder
from NNVisBuilder.utils import tsne
from torch_geometric.utils import subgraph
from torch_geometric.data import Data as Data1
import random
from kmeans_pytorch import kmeans
import argparse

dataset_cora = Planetoid(root='./cora/', name='Cora')
print(dataset_cora[0])
new_data = dataset_cora[0]

parser = argparse.ArgumentParser()
parser.add_argument('--nt', action='store_false', dest='test', help='specify to run not test')
if_test = parser.parse_args().test

# selected_nodes = random.sample(range(2708), 256)
#
# edge_index, _ = subgraph(selected_nodes, dataset_cora.data.edge_index, relabel_nodes=True)
# new_x = dataset_cora.data.x[selected_nodes]
# new_y = dataset_cora.data.y[selected_nodes]
# new_train_mask = dataset_cora.data.train_mask[selected_nodes]
# new_val_mask = dataset_cora.data.val_mask[selected_nodes]
# new_test_mask = dataset_cora.data.test_mask[selected_nodes]
#
# new_data = Data1(x=new_x, edge_index=edge_index, y=new_y, train_mask=new_train_mask,
#                  val_mask=new_val_mask, test_mask=new_test_mask)


# data = dataset_cora[0]
# G = to_networkx(data)
# nodes = torch.randperm(data.num_nodes)[:100]
# subgraph = nx.subgraph(G, nodes.tolist())
# dataset_cora = data.subgraph(nodes)
# dataset_cora = Planetoid(root='./citeseer',name='Citeseer')
# dataset_cora = Planetoid(root='./pubmed/',name='Pubmed')

# Define the teacher model (GCN)
class GCNNet(nn.Module):
    def __init__(self):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(dataset_cora.num_features, 16)
        self.conv2 = GCNConv(16, dataset_cora.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class GCNNet1(nn.Module):
    def __init__(self):
        super(GCNNet1, self).__init__()
        self.conv1 = GCNConv(dataset_cora.num_features, 32)
        self.conv2 = GCNConv(32, dataset_cora.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


# Define the student model (GAT)
class GATNet(nn.Module):
    def __init__(self):
        super(GATNet, self).__init__()
        self.conv1 = GATConv(dataset_cora.num_features, 8, heads=8, dropout=0.6)
        self.conv2 = GATConv(8 * 8, dataset_cora.num_classes, dropout=0.6)

    def forward(self, data):
        x = F.dropout(data.x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, data.edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, data.edge_index)
        return F.log_softmax(x, dim=1)


# Define the student model (SAGEConv)
class SAGENet(nn.Module):
    def __init__(self):
        super(SAGENet, self).__init__()
        self.conv1 = SAGEConv(dataset_cora.num_node_features, 16)
        self.conv2 = SAGEConv(16, dataset_cora.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


# Define the distillation loss function
class DistillKL(nn.Module):
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, outputs, teacher_outputs):
        outputs = F.log_softmax(outputs / self.T, dim=1)
        teacher_outputs = F.softmax(teacher_outputs / self.T, dim=1)
        loss = nn.KLDivLoss()(outputs, teacher_outputs) * self.T * self.T
        return loss


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Instantiate the models and move them to the device
teacher_model1 = GCNNet().to(device)
# teacher_model2 = GCNNet1().to(device)
student_model1 = GATNet().to(device)
student_model2 = SAGENet().to(device)
models = [teacher_model1, student_model1, student_model2]

# Load the data and move it to the device
data = new_data.to(device)

# Define the loss functions and optimizer for the models
criterion_teacher1 = nn.CrossEntropyLoss().to(device)
# criterion_teacher2 = nn.CrossEntropyLoss().to(device)
criterion_student1 = nn.CrossEntropyLoss().to(device)
criterion_student2 = nn.CrossEntropyLoss().to(device)
distill_loss1 = DistillKL(T=2).to(device)
distill_loss0 = DistillKL(T=2).to(device)
distill_loss2 = DistillKL(T=2).to(device)
optimizer_teacher1 = optim.Adam(teacher_model1.parameters(), lr=0.01, weight_decay=5e-4)
# optimizer_teacher2 = optim.Adam(teacher_model2.parameters(), lr=0.01, weight_decay=5e-4)
optimizer_student1 = optim.Adam(student_model1.parameters(), lr=0.01, weight_decay=5e-4)
optimizer_student2 = optim.Adam(student_model2.parameters(), lr=0.01, weight_decay=5e-4)

# Train the models with distillation
for model in models:
    model.train()
builder_t1 = Builder(teacher_model1)
# builder_t2 = Builder(teacher_model2)
builder_s1 = Builder(student_model1)
builder_s2 = Builder(student_model2)
builders = [builder_t1, builder_s1, builder_s2]
for builder in builders:
    builder.add_hiddens(['conv1', 'conv2'], activate=False)
acc = [[], [], []]
preds = [[], [], []]

n_epoch = 200
if if_test:
    n_epoch = 100
for epoch in range(n_epoch):
    # Train the teacher model
    optimizer_teacher1.zero_grad()
    out_teacher1 = teacher_model1(data)
    loss_teacher1 = criterion_teacher1(out_teacher1[data.train_mask], data.y[data.train_mask])
    loss_teacher1.backward()
    optimizer_teacher1.step()

    # Train the student models with distillation
    optimizer_student1.zero_grad()
    out_teacher1 = teacher_model1(data)
    out_student1 = student_model1(data)
    loss_student1 = criterion_student1(out_student1[data.train_mask], data.y[data.train_mask])
    distill_loss1_ = distill_loss1(out_student1[data.train_mask], out_teacher1[data.train_mask])
    loss_student1 += distill_loss1_
    loss_student1.backward()
    optimizer_student1.step()

    optimizer_student2.zero_grad()
    out_teacher = teacher_model1(data)
    out_student2 = student_model2(data)
    loss_student2 = criterion_student2(out_student2[data.train_mask], data.y[data.train_mask])
    distill_loss = distill_loss2(out_student2[data.train_mask], out_teacher[data.train_mask])
    loss_student2 += distill_loss
    loss_student2.backward()
    optimizer_student2.step()

    # Calculate and print the training accuracy and loss for each model
    _, pred_teacher1 = torch.max(out_teacher1[data.train_mask], dim=1)
    correct_teacher1 = (pred_teacher1 == data.y[data.train_mask]).sum().item()
    acc_teacher1 = correct_teacher1 / data.train_mask.sum().item()
    print('Epoch {:03d} Teacher1: train_loss: {:.4f} train_acc: {:.4f}'.format(
        epoch, loss_teacher1.item(), acc_teacher1))
    acc[0].append(acc_teacher1)
    _, pred_student1 = torch.max(out_student1[data.train_mask], dim=1)
    correct_student1 = (pred_student1 == data.y[data.train_mask]).sum().item()
    acc_student1 = correct_student1 / data.train_mask.sum().item()
    print('Epoch {:03d} Student1: train_loss: {:.4f} train_acc: {:.4f}'.format(
        epoch, loss_student1.item(), acc_student1))
    acc[1].append(acc_student1)
    _, pred_student2 = torch.max(out_student2[data.train_mask], dim=1)
    correct_student2 = (pred_student2 == data.y[data.train_mask]).sum().item()
    acc_student2 = correct_student2 / data.train_mask.sum().item()
    print('Epoch {:03d} Student2: train_loss: {:.4f} train_acc: {:.4f}'.format(
        epoch, loss_student2.item(), acc_student2))
    acc[2].append(acc_student2)
    if epoch % 50 == 49:
        # # Evaluate the student models on the test set
        for builder in builders:
            builder.activate_hooks()
        for model in models:
            model.eval()
        out_teacher1 = teacher_model1(data)
        _, pred_teacher1 = torch.max(out_teacher1, dim=1)
        preds[0].append(pred_teacher1.cpu())
        out_student1 = student_model1(data)
        _, pred_student1 = torch.max(out_student1, dim=1)
        preds[1].append(pred_student1.cpu())
        out_student2 = student_model2(data)
        _, pred_student2 = torch.max(out_student2, dim=1)
        preds[2].append(pred_student2.cpu())
        for model in models:
            model.train()
        for builder in builders:
            builder.deactivate_hooks()
for builder in builders:
    for k in builder.embeddings:
        builder.embeddings[k] = builder.embeddings[k].reshape([-1, new_data.num_nodes, builder.embeddings[k].shape[1]])
n_steps = builders[0].embeddings['conv1'].shape[0]


def var1(data):
    return np.var(data, axis=0)


def avg1(data):
    return np.average(data, axis=0)


def km(data):
    return kmeans(data, dataset_cora.num_classes, device=device)[0]


if if_test:
    def tsne(data):
        return data[:, :2]


    def km(data):
        return list(range(data.shape[0]))


embedding = {
    't1': {
        'conv1': builder_t1.embeddings['conv1'],
        'conv2': builder_t1.embeddings['conv2'],
    },
    # 't2': {
    #     'conv1': builder_t2.embeddings['conv1'],
    #     'conv2': builder_t2.embeddings['conv2'],
    # },
    's1': {
        'conv1': builder_s1.embeddings['conv1'],
        'conv2': builder_s1.embeddings['conv2'],
    },
    's2': {
        'conv1': builder_s2.embeddings['conv1'],
        'conv2': builder_s2.embeddings['conv2'],
    },
}

layers = ['conv1', 'conv2']
# based on the order of dict
model_keys = list(embedding.keys())
tls = ['t1', 't2']
ls = ['l1', 'l2']

for j in range(len(model_keys)):
    k = model_keys[j]
    for i in range(2):
        k1 = layers[i]
        k2 = tls[i]
        k3 = ls[i]
        embedding[k][k2] = np.stack([tsne(embedding[k][k1][j]) for j in range(n_steps)])
        embedding[k][k3] = np.stack([km(embedding[k][k1][j]) for j in range(n_steps)])
    embedding[k]['pred'] = np.stack(preds[j])

data1s = []
r_data1s = []
r_data2s = []
data1_s = []
datat1s = []
datat1_s = []
data2s = []
filters = []
filter_ranges = []
fs = []
sp1s = []
sp2s = []
bcs = []
ros = []
dvs = []
das = []
label_kms = []
ground_truth = Data(data.y)
labels = []
label_rs = []
label2s = []
filter_step = Filter(value=n_steps - 1)
filter1 = Filter(value=list(range(new_data.num_nodes)))

select_step = Slider([300, 40], [360, 20], n_steps, title='step:', default_value=n_steps-1)


def f(value):
    filter_step.update(value)


select_step.onclick(f)
label_type = ['ground', 'pred', 'kmeans']
select_comp1 = Select([100, 100], [20, 20], label_type, title='label comparison:', title_vertical=True)
select_comp2 = Select([100, 100], [20, 20], label_type, title='and', title_vertical=True)
select_layer = Select([120, 30], [20, 20], layers, title='layer:')
select_cm = Select([0, 0], [20, 20], ['near', 'label'], title='match mode:', title_vertical=True)


def get_handler(i):
    def f(value):
        if value == -1:
            # for j in range(len(model_keys)):
            #     # filters[j].update([])
            #     sp1s[j].highlighter.update([[], [], sp1s[j].highlighter.value[2]])
            # filters[i].update([])
            sp1s[i].highlighter.update([[], [], [], []])
            filter1.update(list(range(r_data1s[0].size()[0])))
        else:
            if select_cm.value == 0:
                selected = r_data1s[i].nearest_k(value, k=100)
            elif select_cm.value == 1:
                selected = label_rs[i].where(label_rs[i][value])
            for j in range(len(model_keys)):
                com1 = Data(data_type=Type.Vector)
                com2 = Data(data_type=Type.Vector)
                if select_comp1.value == 0:
                    com1.update(ground_truth)
                elif select_comp1.value == 1:
                    com1.update(embedding[model_keys[j]]['pred'][filter_step.value_()])
                elif select_comp1.value == 2:
                    com1.update(embedding[model_keys[j]][ls[select_layer.value]][filter_step.value_()])
                if select_comp2.value == 0:
                    com2.update(ground_truth)
                elif select_comp2.value == 1:
                    com2.update(embedding[model_keys[j]]['pred'][filter_step.value_()])
                elif select_comp2.value == 2:
                    com2.update(embedding[model_keys[j]][ls[select_layer.value]][filter_step.value_()])
                s1 = np.where(com1.value_() == com2.value_())[0].tolist()
                # s1 = com1.where(com1[value])
                # s2 = com2.where(com2[value])
                sp1s[j].highlighter.update([s1, [], selected, []])
                # sp1s[j].highlighter.update(selected)
                filters[j].update(selected)
            filter1.update(selected)

    return f


bg_colors = ['#e41a1c', '#4daf4a','#984ea3']
titles = ['Teacher-1', 'Student-1', 'Student-2']

bg_stroke_opacity = 0.6
bg_stroke_color = 'gray'
bg_stroke_width = 2.6
bg_opacity = 0.6
bg1 = View(None, [165, 25], [1440, 35], color='#EBEBEB', stroke_width=bg_stroke_width, stroke_color=bg_stroke_color, stroke_opacity=bg_stroke_opacity, opacity=bg_opacity)
select_layer.set_position(bg1.align('right(60), under(5)'))
select_step.set_position(select_layer.align('right(500)'))
backgrounds = []

for i, k in enumerate(model_keys):
    # if i > 1:
    #     break
    data1_ = Data(embedding[k]['conv1'])
    datat1_ = Data(embedding[k]['t1'])
    data1_s.append(data1_)
    datat1_s.append(datat1_)
    data1 = data1_.apply_transform(filter_step)
    datat1 = datat1_.apply_transform(filter_step)
    data1s.append(data1)
    datat1s.append(datat1)
    filter = Filter(filter_type=Type.Vector)
    filters.append(filter)
    filter_range = Filter(value=list(range(data1.size()[0])))
    filter_ranges.append(filter_range)
    sp1_data = datat1.apply_transform(filter_range)
    r_data1 = data1.apply_transform(filter_range)
    r_data1s.append(r_data1)
    data2 = r_data1.apply_transform(filter)
    data2s.append(data2)
    label = Data(ground_truth)
    # label1 = Data(ground_truth)
    # label2 = Data(embedding[k]['pred'])
    labels.append(label)
    label = label.apply_transform(filter_range)
    label_rs.append(label)
    background = View(None, [360*i+165, 60], [360, 900], stroke_width=bg_stroke_width, stroke_color=bg_stroke_color, stroke_opacity=bg_stroke_opacity, opacity=bg_opacity)
    backgrounds.append(background)
    # ----------
    reorder = Reorder()
    ros.append(reorder)
    r_data2 = r_data1.apply_transform(filter1)
    r_data2s.append(r_data2)
    dv = r_data2.apply_transform(OtherTransform(var1))
    da = r_data2.apply_transform(OtherTransform(avg1))
    dvs.append(dv)
    das.append(da)
    bc = PointChart(dv.apply_transform(reorder), [360 * i + 200, 100], [300, 200], e_idx=reorder,
                    highlighter=HighLighter(type=Type.Scalar, style='pc'), data1=da.apply_transform(reorder))
    bcs.append(bc)
    sp1 = ScatterPlot(sp1_data, bc.align('under(30, next)'), [300, 300], color_labels=label, save_range=True,
                      cm=cmaps['q7'], highlighter=MultiHighLighter('sp_m'), point_size=1.5, border_color='white')
    sp1s.append(sp1)
    sub_labels = label.apply_transform(filter)
    sp2 = ScatterPlot(data2.apply_transform(OtherTransform(tsne)), sp1.align('under(30, next)'), [300, 295],
                      color_labels=sub_labels, border_color='white', highlighter=HighLighter('circle_size1', type=Type.Scalar),
                      cm=cmaps['q7'], point_size=2.6)
    def g(value):
        sp1.highlighter.update(filter.value_()[value], 3)
    sp2.highlighter.add_mapping(g)
    sp2s.append(sp2)
    sp1.onclick(get_handler(i))
    title = Title(titles[i], background.align(), [360, 30], bg_color='#D5D5D5', color='#656565')
    title.position[0] += 1
    title.size[0] -= 2
    line1 = Line([10, 10], background.align(), color='gray', width=2, opacity=0.6)
    line2 = Line([10, 10], background.align(), color='gray', width=2, opacity=0.6)
    line1.position[1] = 320
    line2.position[1] = 640
bg2 = View(None, bcs[0].align('right(-120), under(-10)'), [120, 230], color='#EBEBEB', stroke_width=bg_stroke_width, stroke_color=bg_stroke_color, stroke_opacity=bg_stroke_opacity, opacity=bg_opacity)
bg3 = View(None, sp1s[0].align('right(-120), under(-10)'), [120, 320], color='#EBEBEB', stroke_width=bg_stroke_width, stroke_color=bg_stroke_color, stroke_opacity=bg_stroke_opacity, opacity=bg_opacity)
bg2.position[0] = bg3.position[0] = backgrounds[0].position[0] - 120
bg4 = View(None, backgrounds[-1].align('right(next)'), [500, 400], stroke_width=bg_stroke_width, stroke_color=bg_stroke_color, stroke_opacity=bg_stroke_opacity, opacity=bg_opacity)
bg5 = View(None, bg4.align('under(next)'), [500, 500], stroke_width=bg_stroke_width, stroke_color=bg_stroke_color, stroke_opacity=bg_stroke_opacity, opacity=bg_opacity)
bg6 = View(None, bg4.align('under(-35)'), [500, 35], color='#EBEBEB', stroke_width=bg_stroke_width, stroke_color=bg_stroke_color, stroke_opacity=bg_stroke_opacity, opacity=bg_opacity)
which_model = 0
which_dim = 0
range_data = Data([-1, 1])
tt = Tooltip1(range_data)


def f(v1, v2):
    r = data1s[which_model].where_range(which_dim, v1, v2)
    for filter in filter_ranges:
        filter.update(r)


tt.onclick(f)
# def f(value):
#     r = data1s[which_model].where_range(which_dim, input1.value, value)
#     for filter in filter_ranges:
#         filter.update(r)
# input2.onclick(f)

select_by = Select(bc.align('right(-100, next)'), [20, 20], ['var', 'avg'], title='by:', title_vertical=True)
# select_by.position[1] = select_layer.position[1]
button_sort = Button(bg2.align('right(10), under(30)'), [60, 20], 'sort')
select_by.set_position(button_sort.align('under(50)'))


def f(value):
    if select_by.value == 0:
        for i in range(len(model_keys)):
            ros[i].update(dvs[i].argsort(reverse=True))
    else:
        for i in range(len(model_keys)):
            ros[i].update(das[i].argsort(reverse=True))


button_sort.onclick(f)

plc = ParallelCoordinate(Data(acc), position=bg5.align('right(60), under(100)'), size=[400, 300], title='Accuracy', threshold=None,
                         colors=bg_colors, legend=[{'label': titles[i], 'color': bg_colors[i]} for i in range(len(titles))])




def f(value):
    for i in range(len(data1s)):
        sp1s[i].save_range = False
        data1_s[i].update(embedding[model_keys[i]][layers[value]])
        datat1_s[i].update(embedding[model_keys[i]][tls[value]])


select_layer.onclick(f)


select_label = Select(bg3.align('right(10), under(30)'), [20, 20], label_type, title='label:', title_vertical=True)
select_comp1.set_position(select_label.align('under(100)'))
select_comp2.set_position(select_comp1.align('under(45)'))
select_cm.set_position(select_comp2.align('under(88)'))


def f(value):
    if value == 0:
        for i in range(len(model_keys)):
            labels[i].update(ground_truth)
    elif value == 1:
        for i in range(len(model_keys)):
            labels[i].update(embedding[model_keys[i]]['pred'][filter_step.value_()])
    elif value == 2:
        for i in range(len(model_keys)):
            labels[i].update(embedding[model_keys[i]][ls[select_layer.value]][filter_step.value_()])


select_label.onclick(f)

hs = [[16, 7], [64, 7], [16, 7]]
select_sim1 = Select(bg6.align('right(180), under(5)'), [36, 20], titles, title='similarity between')
select_sim2 = Select(select_sim1.align('right(123)'), [36, 20], titles, title='and')
# select3 = Select([1500, 450], [20, 20], ['var', 'avg'])
#
#
# def f(value):
#     if value == 0:
#         for data in data1s:
#             data.named_filters['tr'].update(var1)
#     elif value == 1:
#         for data in data1s:
#             data.named_filters['tr'].update(avg1)
#
#
# select3.onclick(f)
data3 = Data(value=[0])
heat_map = HeatMap(data3, position=bg4.align('right(30), under(25)'), cell_size=[25*1.1, 20*1.1], cm='Blues', highlighter=MultiHighLighter('hm_m'), selector='pos')
def f(row, col):
    bcs[select_sim1.value].highlighter.update(row)
    bcs[select_sim2.value].highlighter.update(col)
    heat_map.highlighter.update([row, col])
heat_map.onclick(f)

def f(value):
    temp = torch.zeros((hs[value][select_layer.value], hs[select_sim2.value][select_layer.value]))
    for i in range(hs[value][select_layer.value]):
        temp[i] = torch.nn.functional.cosine_similarity(
            embedding[model_keys[value]][layers[select_layer.value]][select_step.value][:, i].reshape(-1, 1),
            embedding[model_keys[select_sim2.value]][layers[select_layer.value]][select_step.value], dim=0)
    data3.update(temp)


select_sim1.onclick(f)


def f(value):
    temp = torch.zeros((hs[select_sim1.value][select_layer.value], hs[value][select_layer.value]))
    for i in range(hs[select_sim1.value][select_layer.value]):
        temp[i] = torch.nn.functional.cosine_similarity(
            embedding[model_keys[select_sim1.value]][layers[select_layer.value]][select_step.value][:, i].reshape(-1, 1),
            embedding[model_keys[value]][layers[select_layer.value]][select_step.value], dim=0)
    data3.update(temp)


select_sim2.onclick(f)


def get_f1(i):
    def f(value):
        bcs[i].highlighter.update(value)
        for sp in sp1s:
            sp.save_range = True
        global which_dim, which_model
        which_model = i
        which_dim = value
        range_data.update([r_data2s[i][:, value].min(), r_data2s[i][:, value].max()])
        if select_sim1.value == i:
            heat_map.highlighter.update(value, 0)
        if select_sim2.value == i:
            heat_map.highlighter.update(value, 1)

    return f


for i in range(len(bcs)):
    bcs[i].onclick(get_f1(i))

builder_t1.run()

# loss = criterion_student(out_student1[data.test_mask], data.y[data.test_mask])
# _, pred = torch.max(out_student1[data.test_mask], dim=1)
# correct = (pred == data.y[data.test_mask]).sum().item()
# acc = correct/data.test_mask.sum().item()
# print("test_loss s1: {:.4f} test_acc: {:.4f}".format(loss.item(), acc))
