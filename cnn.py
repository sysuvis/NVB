import json
import os
import random

import torch
import torchvision
import torchvision.transforms as transforms
# 这份代码使用了github上的一个开源项目，如果侵权的话，后续我会进行替换
########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].

########################################################################
# .. note::
#     If running on Windows and you get a BrokenPipeError, try setting
#     the num_worker of torch.utils.data.DataLoader() to 0.

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 128



trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                   shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                  shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

########################################################################
# Let us show some of the training images, for fun.

import matplotlib.pyplot as plt
import numpy as np


# functions to show an image


def imsave(img):
    img = img / 2 + 0.5  # unnormalize
    print(type(img))
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# # show images
# imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

########################################################################
# 2. Define a Convolutional Neural Network
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Copy the neural network from the Neural Networks section before and modify it to
# take 3-channel images (instead of 1-channel images as it was defined).

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

net = Net().to(device)

########################################################################
# 3. Define a Loss function and optimizer
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Let's use a Classification Cross-Entropy loss and SGD with momentum.

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)



PATH = 'model/cifar_net.pth'
# torch.save(net.state_dict(), PATH)
# exit()

# dataiter = iter(testloader)
# images, labels = dataiter.next()



# ln = [0 for i in range(10)]
# # # print images
# a = images.numpy().transpose(0, 2, 3, 1)
# for i in range(a.shape[0]):
#     t = a[i] / 2 + 0.5
#     plt.imsave('static/img/%d.png' % i, t)
#     if ln[labels[i]] < 4:
#         plt.imsave('static/img/am/%d_%d.png'%(labels[i], ln[labels[i]]), t)
#         ln[labels[i]] += 1

# a *= 255
# print(a)
# from PIL import Image
# a = Image.fromarray(a.transpose(1,2,0), mode='RGB')
# a.show()
# exit()
# print(images.numpy().shape)
# exit()
# imsave(a)
# print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

########################################################################
# Next, let's load back in our saved model (note: saving and re-loading the model
# wasn't necessary here, we only did it to illustrate how to do so):

net = Net().to(device)
net.load_state_dict(torch.load(PATH))
print('Loaded.')
from NNVisBuilder.Data import *
from sklearn.manifold import TSNE

from NNVisBuilder import Builder
from NNVisBuilder.Views.Views import *
from NNVisBuilder.Views.Widgets import *
# builder = Builder(net, input_=images, targets=np.array(labels))
# print(builder.name2module['conv1'].out_channels, builder.name2module['conv1'].in_channels)
# # exit()
# layers = ['conv1', 'pool', 'conv2', 'fc1', 'fc2', 'fc3']
# builder.add_hiddens(layers)


import random

cifar10_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# 按类别进行划分
classes = cifar10_train.classes
data_by_class = {c: [] for c in classes}
for i in range(len(cifar10_train)):
    image, label = cifar10_train[i]
    data_by_class[classes[label]].append((image, label))

# 针对每个类别，随机选择前10个数据
selected_data = []
for c in classes:
    data = data_by_class[c]
    selected = random.sample(data, 10)
    selected_data.extend(selected)

images = [x[0] for x in selected_data]
labels = [x[1] for x in selected_data]
images = np.stack(images, axis=0)

# for i, (image, label) in enumerate(selected_data):
#     print(f"Sample {i+1}, Class: {classes[label]}")


# ln = [0 for i in range(10)]
# # print images
a = images.transpose(0, 2, 3, 1)
os.chdir('NNVisBuilder')
for i in range(a.shape[0]):
    builder = a[i] / 2 + 0.5
    plt.imsave('static/img/%d.png' % i, builder)
    # # 这是个啥，我自己都不记得了
    # if ln[labels[i]] < 4:
    #     plt.imsave('static/img/am/%d_%d.png'%(labels[i], ln[labels[i]]), t)
    #     ln[labels[i]] += 1


images = torch.from_numpy(images)
images = images.to(device)

builder = Builder(net, input_=images, targets=np.array(labels))
print(builder.name2module['conv1'].out_channels, builder.name2module['conv1'].in_channels)
# exit()
layers = ['conv1', 'conv2', 'fc1', 'fc2', 'fc3']
builder.add_hiddens(layers)
num_class = 10
sub_size = 10

builder.forward(batch_size=100)

t_data = Data(value=list(range(sub_size)))
t_prefix = Data(data_type=Type.Scalar)
toolTip = Tooltip(t_data, prefix=t_prefix)

class_filter = Filter(dim=0, value=0)
for i in range(2):
    layer = layers[i]
    embedding = builder.embeddings[layer]
    for j in range(builder.name2module[layer].out_channels):
        for c in range(num_class):
            pics = Data(embedding[c*10:(c+1)*10, j, :, :])
            for k in range(sub_size):
                Data(pics[k]).save_img(k, prefix='%s-%d-%d' % (layer, j, c))
            pics.aggregation(op='max').data().save_img(c, prefix='%s-%d' % (layer, j))
        if j == 0:
            pic_view = Picture(class_filter, [i * 40 + 120, j * 32 + 250], [30, 30], prefix='%s-%d' % (layer, j), title=layer)
        else:
            pic_view = Picture(class_filter, [i * 40 + j//10*32 + 120, (j %10)* 32 + 250], [30, 30], prefix='%s-%d' % (layer, j))
        def f(value, position):
            t_prefix.update('%s-%d' % (value, class_filter.value_()))
            toolTip.set_position(position)
        pic_view.onclick(f)

h = HighLighter(style='circle_size')
for i in range(2, 4):
    embedding = builder.embeddings[layers[i]]
    data = Data(TSNE().fit_transform(embedding))
    sp = ScatterPlot(data, [i*110+80, 250], [100, 100], color_labels=labels, cm=cmaps['d10'], highlighter=h, title=layers[i])



# reshape to extract a class dimension
embedding = torch.softmax(builder.embeddings[layers[-1]], dim=1).reshape(num_class, sub_size, -1)

img_pos = Data(TSNE().fit_transform(images.cpu().view(num_class*sub_size, -1))).reshape((num_class, sub_size, -1))
for i in range(embedding.shape[2]):
    pos = img_pos.apply_transform(class_filter)
    labels = Data(embedding[:, :, i]).apply_transform(class_filter)
    if i == 0:
        sp = ScatterPlot(pos, [550, 250+64*i], [60, 60], color_labels=labels, point_size=1.5, title='fc3', cm=cmaps['s2'], opacity=1)
    else:
        sp = ScatterPlot(pos, [550+i//5*64, 250 + 64 * (i%5)], [60, 60], color_labels=labels, point_size=1.5, cm=cmaps['s2'], opacity=1)
select = Select([120, 180], options=classes)
def f(value):
    class_filter.update(value)
select.onclick(f)

g_data = Data(data_type=Type.Vector)
gallery = Gallery(g_data, [300, 360], [210, 500])
def f(value):
    g_data.update(value)
h.add_mapping(f)




# d1 = Data(data_type=DataType.Matrix)
# h1 = HighLighter(style=)
# v1 = ParallelCoordinate(d1, position=[100, 100], size=[500, 200])

# d2 = Data(data_type=Type.Matrix)
# # h2 = HighLighter()
# h2 = HighLighter(style='circle_size')
# v2 = ScatterPlot(d2, position=[100, 350], size=[200, 200], highlighter=h2)
# d3 = Data(data_type=Type.Vector)
# v3 = Gallery(d3, position=v2.align('right(100, next)'))
# d2.update(TSNE().fit_transform(builder.embeddings['fc1']))
#
#
# def f(value):
#     h2.update([i for i in range(20)])
#     d3.update(['%d.png'%i for i in range(20)])
#
#
# v2.onclick(f)
# print(View.update_list)

builder.run(foward_manual=True)




