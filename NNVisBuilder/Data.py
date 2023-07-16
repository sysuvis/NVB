import numpy as np
from enum import Enum
import torch
from PIL import Image
from NNVisBuilder.Views import View
from NNVisBuilder.GlobalVariables import *


class Type(Enum):
    Scalar = 0
    Vector = 1
    Matrix = 2
    Tensor3 = 3
    Tensor4 = 4
    # perhaps no need
    Tensor5 = 5


class Data:
    def __init__(self, value=None, data_type=None, builder=None, info=None):
        """
        If value is list, type will be Vector.
        Elif value is tensor or ndarray, type will be the order of data.
        Else type will be Scalar.
        builder and info will not be used, and will be deleted later
        """
        self.type = data_type
        if value is None:
            if data_type == Type.Scalar:
                self.value = 0
            else:
                self.value = np.zeros([1] * data_type.value)
        else:
            if not isinstance(value, (list, torch.Tensor, np.ndarray)):
                self.type = Type.Scalar
            elif isinstance(value, list):
                self.type = Type.Vector
            else:
                self.type = Type(len(value.shape))
            if isinstance(value, (list, torch.Tensor)):
                if isinstance(value, torch.Tensor):
                    value = value.cpu()
                self.value = np.array(value, copy=isinstance(value, list))
            else:
                self.value = value
        self.views = []
        self.named_filters = {}
        self.rules = []
        self.info = {}
        if builder is not None:
            self.builder = builder
            self.info = info

    def filter(self, dim=0, filter_type=Type.Scalar, name=None, value=None):
        t = Filter(dim, filter_type, value=value, data=self, root_data=self)
        self.rules.append(t)
        if name is not None:
            self.named_filters[name] = t
        return t

    def aggregation(self, dim=0, op='sum'):
        t = Aggregation(dim, op, data=self, root_data=self)
        self.rules.append(t)
        return t

    def reshape(self, shape=(-1,)):
        if not isinstance(shape, (list, tuple)):
            shape = (shape,)
        t = Reshape(shape, data=self, root_data=self)
        self.rules.append(t)
        # why return data here? because of the practice
        return t.data()
        # return t

    def save_img(self, ids, prefix='', suffix='png'):
        value = (self.value - np.min(self.value)) / (np.max(self.value) - np.min(self.value))
        image = Image.fromarray(value.astype('float32'), 'L')
        image.save('static/img/' + prefix + '-' + str(ids) + '.' + suffix)

    def sign(self):
        t = OtherTransform(np.sign, data=self, root_data=self)
        self.rules.append(t)
        return t.data()

    def max(self, axis=None):
        # max currently not in dynamic
        return np.max(self.value, axis)

    def min(self, axis=None):
        # max currently not in dynamic
        return np.min(self.value, axis)

    def other_transform(self, f):
        t = OtherTransform(f, data=self, root_data=self)
        self.rules.append(t)
        return t

    def apply_transform(self, t, dim=-1, name=None, flag=True):
        # non-minus1 dim will cover default of t
        if name is not None and isinstance(t, (Filter, OtherTransform)):
            self.named_filters[name] = t
        self.rules.append(t)
        return t.apply_transform(self, dim, flag)

    def update(self, value, flag=True):
        if isinstance(value, Data):
            value = value.value_()
        if isinstance(value, (list, torch.Tensor)):
            if isinstance(value, torch.Tensor):
                value = value.cpu()
            value = np.array(value, copy=isinstance(value, list))
        self.value = value
        if flag:
            self.update_()
        return self

    def update_(self):
        for view in self.views:
            if view.idx not in View.update_list:
                View.update_list.append(view.idx)
        for rule in self.rules:
            # include update_ of result_data
            rule.apply_data(self)

    def value_(self):
        return self.value

    def size(self):
        return self.value.shape

    def where(self, x):
        return np.where(self.value == x)[0].tolist()

    def argsort(self, dim=0, reverse=False):
        # handle dim ...
        if reverse:
            return np.argsort(-self.value)
        else:
            return np.argsort(self.value)

    def where_range(self, d, a, b):
        # temporally used function
        i_mask = (self.value[:, d] >= a) & (self.value[:, d] <= b)
        return np.where(i_mask)[0].tolist()

    def tolist(self):
        # reshape(-1) should be seperated
        return self.value.reshape(-1).tolist()

    def __getitem__(self, index):
        return self.value[index]

    def __setitem__(self, index, value):
        self.value[index] = value
        self.update_()

    def __len__(self):
        return len(self.value)

    def nearest_k(self, idx, k=10):
        dists = np.array([np.linalg.norm(self.value[i] - self.value[idx]) for i in range(self.value.shape[0])])
        return dists.argsort()[:k].tolist()

    def find_all(self, f, dim=0):
        r = []
        if dim == 0:
            for i in range(self.value.shape[dim]):
                if f(self.value[i]):
                    r.append(i)
        elif dim == 1:
            for i in range(self.value.shape[dim]):
                if f(self.value[:, i]):
                    r.append(i)
        return r


class Rule:
    def __init__(self, dim=0, data=None, root_data=None):
        # dim: default dim
        self.dim = dim
        self.source_data = [data] if data is not None else []
        # dims: specific dim for each source_data
        self.dims = [self.dim] if data is not None else []
        self.root_data = root_data
        self.result_data = []

    def data(self):
        # maybe restriction here and just for series construction
        return self.result_data[0]

    def filter(self, dim=0, filter_type=Type.Scalar, name=None, value=None):
        t = Filter(dim, filter_type, value=value, data=self.data(), root_data=self.root_data)
        self.data().rules.append(t)
        if name is not None and self.root_data is not None:
            self.root_data.named_filters[name] = t
        return t

    def aggregation(self, dim=0, op='sum'):
        t = Aggregation(dim, op, data=self.data(), root_data=self.root_data)
        self.data().rules.append(t)
        return t

    def reshape(self, shape=(-1,)):
        t = Reshape(shape, data=self.data(), root_data=self.root_data)
        self.data().rules.append(t)
        return t

    def other_transform(self, f):
        t = OtherTransform(f, data=self.data(), root_data=self.root_data)
        self.data().rules.append(t)
        return t

    def apply_transform(self, data, dim=-1, flag=True):
        self.source_data.append(data)
        self.dims.append(dim if dim != -1 else self.dim)
        return self.generate_result(data, flag)

    def generate_result(self, data, flag=True):
        # generate a Data object for result, flag=True call apply_i to compute the value of it
        pass

    def apply_i(self, i, flag=True):
        # compute the result for the ith source_data
        pass

    def apply(self):
        for i in range(len(self.source_data)):
            self.apply_i(i, False)

    def apply_data(self, data):
        i = self.source_data.index(data)
        self.apply_i(i, True)
        return i


class Filter(Rule):
    def __init__(self, dim=0, filter_type=None, value=None, data=None, root_data=None):
        """
        :param dim:
        :param filter_type:
        :param value:
        :param data: User are not recommended to use this param, because no data.rule.append(self) here.
        :param root_data:
        """
        super(Filter, self).__init__(dim, data, root_data=root_data)
        self.type = filter_type
        if filter_type is None:
            if value is not None:
                self.type = Type(len(np.array(value).shape))
            else:
                self.type = Type.Scalar
        if self.type == Type.Scalar:
            self.filter_value = 0
        elif self.type == Type.Vector:
            self.filter_value = [0]
        elif self.type == Type.Matrix:
            self.filter_value = [[0]]
        # filter_value is list not numpy array
        if len(self.source_data) == 1:
            self.generate_result(self.source_data[0], False)
        self.views = []
        if value is not None:
            self.update(value)

    def from_1d(self, value, steps, max_len):
        if not isinstance(value, Data):
            r = []
            for i in value:
                r.append([])
                for j in range(steps):
                    if i + j < max_len:
                        r[-1].append(i + j)
            self.update(r)
        else:
            pass  # TODO

    def update(self, value, flag=True):
        if self.type == Type.Vector and not isinstance(value, list):
            if value in self.filter_value:
                self.filter_value.remove(value)
            else:
                self.filter_value.append(value)
        else:
            self.filter_value = value
            if self.type == Type.Matrix and len(value) == 0:
                self.filter_value = [[]]
        self.apply()
        if flag:
            for view in self.views:
                if view.idx not in View.update_list:
                    View.update_list.append(view.idx)
            for data in self.result_data:
                data.update_()
        return self

    def apply_i(self, i, flag=True):
        if self.type != Type.Matrix:
            s = ':,' * self.dims[i] + str(self.filter_value)
            # another choice: use if else for each case to replace eval
            # optimized data flow can remove this try catch
            try:
                self.result_data[i].value = eval('self.source_data[i].value[%s]' % s)
                # print('a', self.source_data[i].value.shape, self.result_data[i].value.shape, self.type, s)
            except:
                pass
        else:
            # self.result_data[i].value = np.zeros([len(self.filter_value), max([len(s) for s in self.filter_value])])
            # for j in range(len(self.filter_value)):
            #     t = self.source_data[i].value[self.filter_value[j]]
            #     self.result_data[i].value[j, :len(t)] =
            r = []
            for s in self.filter_value:
                r.append(self.source_data[i].value[s].tolist())
            # align the 2d list
            if len(r) != 0 and len(r[0]) != 0:
                l = max(map(len, r))
                t = '' if isinstance(r[0][0], str) else 0
                r = [a + [t] * (l - len(a)) for a in r]
            self.result_data[i].value = np.array(r)
        if flag:
            self.result_data[i].update_()

    def generate_result(self, data, flag=True):
        if self.type == Type.Scalar:
            self.result_data.append(Data(data_type=Type(data.type.value - 1)))
        elif self.type == Type.Vector:
            self.result_data.append(Data(data_type=data.type))
        elif self.type == Type.Matrix:
            self.result_data.append(Data(data_type=Type.Matrix))
        if flag:
            self.apply_i(-1)
        return self.result_data[-1]

    def value_(self):
        return self.filter_value

    def __getitem__(self, index):
        return self.filter_value[index]


class Aggregation(Rule):
    def __init__(self, dim=0, op='sum', data=None, root_data=None, flag=True):
        super(Aggregation, self).__init__(dim, data, root_data=root_data)
        self.op = op
        if len(self.source_data) == 1:
            self.generate_result(self.source_data[0], flag)

    def apply_i(self, i, flag=True):
        if self.op == 'sum':
            self.result_data[i].value = self.source_data[i].value.sum(axis=self.dims[i])
        elif self.op == 'max':
            self.result_data[i].value = self.source_data[i].value.max(axis=self.dims[i])
        elif self.op == 'avg':
            self.result_data[i].value = self.source_data[i].value.mean(axis=self.dims[i])
        if flag:
            self.result_data[i].update_()

    def generate_result(self, data, flag=True):
        self.result_data.append(Data(data_type=Type(data.type.value - 1)))
        if flag:
            self.apply_i(-1)
        return self.result_data[-1]


class Reshape(Rule):
    def __init__(self, shape=(-1,), data=None, root_data=None, flag=True):
        super(Reshape, self).__init__(0, data, root_data=root_data)
        self.shape = shape
        if len(self.source_data) == 1:
            self.generate_result(self.source_data[0], flag)

    def apply_i(self, i, flag=True):
        self.result_data[i].value = self.source_data[i].value.reshape(self.shape)
        if flag:
            self.result_data[i].update_()

    def generate_result(self, data, flag=True):
        self.result_data.append(Data(data_type=Type(len(self.shape))))
        if flag:
            self.apply_i(-1)
        return self.result_data[-1]


class Reorder(Rule):
    def __init__(self, order=None, data=None, root_data=None, flag=True):
        super(Reorder, self).__init__(0, data, root_data=root_data)
        self.order = order
        self.views = []
        if len(self.source_data) == 1:
            self.generate_result(self.source_data[0], flag)

    def update(self, value, flag=True):
        self.order = value
        self.apply()
        if flag:
            for view in self.views:
                if view.idx not in View.update_list:
                    View.update_list.append(view.idx)
            for data in self.result_data:
                data.update_()
        return self

    def value_(self):
        return self.order

    def apply_i(self, i, flag=True):
        if self.order is None:
            self.result_data[i].value = self.source_data[i].value
        else:
            self.result_data[i].value = self.source_data[i].value[self.order]
        if flag:
            self.result_data[i].update_()

    def generate_result(self, data, flag=True):
        self.result_data.append(Data(data_type=Type(len(np.array(self.order).shape))))
        if flag:
            self.apply_i(-1)
        return self.result_data[-1]


class OtherTransform(Rule):
    def __init__(self, f, data=None, root_data=None):
        super(OtherTransform, self).__init__(0, data, root_data=root_data)
        self.f = f
        if len(self.source_data) == 1:
            self.generate_result(self.source_data[0], False)

    def update(self, f, flag=True):
        self.f = f
        self.apply()
        if flag:
            for data in self.result_data:
                data.update_()
        return self

    def apply_i(self, i, flag=True):
        self.result_data[i].value = self.f(self.source_data[i].value)
        if flag:
            self.result_data[i].update_()

    def generate_result(self, data, flag=True):
        self.result_data.append(Data(value=self.f(data.value)))
        if flag:
            self.apply_i(-1)
        return self.result_data[-1]


class Transpose(Rule):
    def __init__(self, data=None, root_data=None):
        super(Transpose, self).__init__(0, data, root_data=root_data)
        if len(self.source_data) == 1:
            self.generate_result(self.source_data[0], False)

    def apply_i(self, i, flag=True):
        self.result_data[i].value = np.transpose(self.source_data[i].value)
        if flag:
            self.result_data[i].update_()

    def generate_result(self, data, flag=True):
        self.result_data.append(Data(value=np.transpose(data.value)))
        if flag:
            self.apply_i(-1)
        return self.result_data[-1]


class HighLighter:
    def __init__(self, style=None, value=None, type=Type.Vector):
        self.style = styles[style] if style is not None else None
        self.value = []
        self.views = []
        self.mappings = []
        self.type = type
        if value is None:
            if type == Type.Scalar:
                self.value = 0
        else:
            self.update(value)

    def update(self, value):
        if self.type == Type.Scalar:
            self.value = value
        else:
            if isinstance(value, np.ndarray):
                value = value.tolist()
            if isinstance(value, list):
                self.value = value
            else:
                if value in self.value:
                    self.value.remove(value)
                else:
                    self.value.append(value)
        for view in self.views:
            if view.idx not in View.highlight_list:
                View.highlight_list.append(view.idx)
        for f in self.mappings:
            if self.type == Type.Scalar:
                f(self.value)
            else:
                f(self.value.copy())

    def add_mapping(self, f):
        self.mappings.append(f)

    def set_style(self, s):
        self.style = s

    def core(self):
        return self.style


# need to be generalized
# (Multi)HighLighter should carry other info in r.
class MultiHighLighter:
    def __init__(self, style, value=None, type=Type.Scalar, n=3):
        self.style = styles[style] if style is not None else None
        self.type = type
        if type == Type.Scalar:
            self.value = [-1, -1]
        elif type == Type.Vector:
            self.value = [[], [], [], []]
        else:
            self.value = [None, None]
        self.n = n
        self.intersect = []
        self.views = []
        self.mappings = []
        if value is not None:
            self.update(value)

    def update(self, value, idx=None):
        if idx is not None:
            self.value[idx] = value
        else:
            self.value = value
            # if len(value) == 3:
            #     self.value = value + []
            # else:
            #     self.value = value
        for view in self.views:
            if view.idx not in View.highlight_list:
                View.highlight_list.append(view.idx)
        for f in self.mappings:
            f(self.value.copy())

    def core(self):
        return self.style
