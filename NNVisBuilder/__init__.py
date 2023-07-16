import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from .Views import View
from .Views.Views import *
from NNVisBuilder.Views.Widgets import *
import torch
from sklearn.manifold import TSNE
from NNVisBuilder.utils import *
from NNVisBuilder.backend import launch
import pandas as pd
from NNVisBuilder.GlobalVariables import *
import webbrowser
import os


class Builder:
    views = []
    widgets = []

    def __init__(self, model=None, input_=None, **info):
        self.model = model
        if model:
            # 没有model的情况是用来方便测试视图的
            for p in model.parameters():
                self.device = p.device
                print('Model device:', self.device)
                break
            self.name2module = dict(model.named_modules())
        self.input = input_
        self.targets = info.get('targets')
        # a heavy bottom to improve
        # self.grads = {}
        self.hooks = {}
        self.hooks_list = []
        self.embeddings = {}
        self.connections = {}
        # for stage 1
        self.hooks_1 = {}
        self.embeddings_1 = {}
        # for decision border(grid data)
        self.hooks_g = {}
        self.hooks_g_list = []
        self.embeddings_g = {}
        self.view_components = []
        self.named_views = {}
        # ....
        self.info = info
        self.ref_pos = [0]
        if info.get('decoder', None):
            self.decoder = info['decoder']
            self.name2decoder_module = dict(self.decoder.named_modules())
            self.sens = {}
        # how to generate gid is based on input range and type
        self.grid = None

    def get_hook(self, name, cat_dim=0, stage=0, f=None):
        embeddings = self.embeddings if stage == 0 else self.embeddings_1
        def hook(model, input_, output):
            embeddings[name] = torch.cat(
                (embeddings[name], (output if f is None else f(output)).detach().cpu()), cat_dim)
        return hook

    def add_connects(self, layers):
        for layer in layers:
            self.connections[layer] = torch.Tensor([])

    def reset_connect(self):
        for layer in self.connections:
            self.connections[layer] = torch.Tensor([])

    def record_connect(self):
        # now only for record weight of linear...
        for layer in self.connections:
            self.connections[layer] = torch.cat((self.connections[layer], self.name2module[layer].weight.detach().unsqueeze(0).cpu()), 0)

    def add_hiddens(self, layers, cat_dim=0, stage=0, component=None, activate=True):
        if not isinstance(layers, list):
            layers = [layers]
        # 但是考虑到可能随着torch版本变化需要，先留着
        # # 这里不需要，可以用内置的，后面删了
        # for layer in layers:
        #     m = self.model
        #     for a in layer.split('-'):
        #         if a.isdigit():
        #             m = list(m)[int(a)]
        #         else:
        #             m = dict(m.named_modules())[a]
        #     self.embeddings[layer] = torch.Tensor([])
        #     self.hooks.append(m.register_forward_hook(get_hook(layer)))
        if stage == 0 or stage == 'all':
            for layer in layers:
                self.embeddings[layer] = torch.Tensor([])
                self.hooks[layer] = self.get_hook(layer, cat_dim, 0, rnn_extract_output if component == 'rnn' else component)
                if activate:
                    self.hooks_list.append(self.name2module[layer].register_forward_hook(self.hooks[layer]))
        if stage == 1 or stage == 'all':
            for layer in layers:
                self.embeddings_1[layer] = torch.Tensor([])
                self.hooks_1[layer] = self.get_hook(layer, cat_dim, 1)

    # def add_view(self, layer_name, view_class, transform=None, position=None, size=None, name=None, **view_info):
    #     if layer_name and layer_name not in self.embeddings:
    #         self.embeddings[layer_name] = torch.Tensor([])
    #     if position is None:
    #         position = [0, 0]
    #     if size is None:
    #         size = [200, 200]
    #     self.view_components.append(ViewComponent(layer_name, view_class, transform, position, size, name, view_info))

    # def add_widget(self, widget_class, position=None, size=None, **widget_info):
    #     self.view_components.append(ViewComponent(None, widget_class, None, position, size, None, widget_info))

    # def composite_view(self, composite_view_info, position=None, size=None):
    #     """
    #     info include the information about which dataframe column is assigned to which visual channel
    #     """
    #     view_info = []
    #     for info in composite_view_info:
    #         view_info.append(info)
    #         for k in info['data']:
    #             t = info['data'][k]
    #             # str implies hidden and need to be extracted after.
    #             if isinstance(t, str):
    #                 if info['element']['class'] == 'decision_border':
    #                     if t not in self.embeddings_g:
    #                         self.embeddings_g[t] = torch.Tensor([])
    #                 else:
    #                     if t not in self.embeddings:
    #                         self.embeddings[t] = torch.Tensor([])
    #     if position is None:
    #         position = [0, 0]
    #     if size is None:
    #         size = [200, 200]
    #     self.view_components.append(ViewComponent(None, CompositeView1, None, position, size, None, view_info))
    #     # self.view_components.append(ViewComponent(None, CompositeView, None, position, size, None, view_info))

    # def add_hook(self, name):
    #     def get_hook(name):
    #         def hook(model, input, output):
    #             self.embeddings[name] = torch.cat(
    #                 (self.embeddings[name], (output[1] if self.info.get('rnn', False) else output).detach().cpu()), 0)
    #
    #         return hook
    #
    #     self.hooks.append(self.name2module[name].register_forward_hook(get_hook(name)))

    def generate_grid_data(self):
        # need improve to fit different ranges params
        n = 16
        input_ = torch.tensor([[x / n, y / n] for x in range(-n, n) for y in range(-n, n)])
        x0, x1 = torch.min(self.input[:, 0]).item(), torch.max(self.input[:, 0]).item()
        y0, y1 = torch.min(self.input[:, 1]).item(), torch.max(self.input[:, 1]).item()
        input_[:, 0] = (input_[:, 0] + 1) / (2 - 1 / n) * (x1 - x0) / 0.95 + x0 - 0.025 / 0.95 * (x1 - x0)
        input_[:, 1] = (input_[:, 1] + 1) / (2 - 1 / n) * (y1 - y0) / 0.95 + y0 - 0.025 / 0.95 * (y1 - y0)
        self.grid = input_
        return Data(value=self.grid.numpy())

    def forward_grid_data(self):
        if self.grid is None:
            print('Please call forward_grid_data() by yourself.')
            return
        for layer in self.hooks_g:
            self.hooks_g_list.append(self.name2module[layer].register_forward_hook(self.hooks_g[layer]))
        i = 0
        c = self.grid.size()[0]
        batch_size = min(32, c)
        with torch.no_grad():
            while batch_size * i < c:
                a = self.grid[batch_size * i:batch_size * (i + 1)]
                self.model(a.to(self.device))
                i += 1
            print('g---------- %d / %d' % (c, c))
        for hook in self.hooks_g_list:
            hook.remove()
        self.hooks_g_list.clear()

    def add_hidden_grid(self, layers):
        for layer in layers:
            self.hooks_g[layer] = self.get_hook_g(layer)
            self.embeddings_g[layer] = torch.Tensor([])

    def get_hook_g(self, name):
        # maybe integrated into get_hook
        def hook(model, input, output):
            self.embeddings_g[name] = torch.cat((self.embeddings_g[name], output.detach().cpu()), 0)
        return hook

    def forward(self, batch_size=32, mode=0):
        # # i really convince
        # for name in self.embeddings:
        #     # it's temp
        #     if name != 'attn':
        #         self.add_hook(name)
        self.model.eval()
        if mode == 1:
            b = 0
            print(len(self.input))
            for s in self.input:
                b += s.shape[0]
                self.ref_pos.append(b)
                hidden = self.model.initHidden()
                for w in s:
                    output, hidden = self.model(w, hidden)
            for k in self.embeddings:
                self.embeddings[k] = self.embeddings[k].squeeze()
            # print(self.ref_pos)
            # exit()
        elif mode == 0:
            i = 0
            c = self.input.size()[0]
            if batch_size > c:
                batch_size = c
            with torch.no_grad():
                while batch_size * i < c:
                    print('---------- %d / %d' % (batch_size * i, c))
                    a = self.input[batch_size * i:batch_size * (i + 1)]
                    self.model(a.to(self.device))
                    i += 1
                print('---------- %d / %d' % (c, c))

    def border_layers(self, layers):
        for layer in layers:
            self.embeddings_g[layer] = torch.Tensor([])

    # def forward_b(self, batch_size=32):
    #     for layer in self.embeddings_g:
    #         self.add_hook_b(layer)
    #         self.embeddings_g[layer] = torch.Tensor([])
    #     if self.grid is None:
    #         n = 32
    #         input_ = torch.tensor([[x / n, y / n] for x in range(-n, n) for y in range(-n, n)])
    #         x0, x1 = torch.min(self.input[:, 0]).item(), torch.max(self.input[:, 0]).item()
    #         y0, y1 = torch.min(self.input[:, 1]).item(), torch.max(self.input[:, 1]).item()
    #         input_[:, 0] = (input_[:, 0] + 1) / (2 - 1 / n) * (x1 - x0) / 0.95 + x0 - 0.025 / 0.95 * (x1 - x0)
    #         input_[:, 1] = (input_[:, 1] + 1) / (2 - 1 / n) * (y1 - y0) / 0.95 + y0 - 0.025 / 0.95 * (y1 - y0)
    #         # t1 = x1 - x0
    #         # t2 = y1 - y0
    #         self.grid = input_
    #     i = 0
    #     c = self.grid.size()[0]
    #     if batch_size > c:
    #         batch_size = c
    #     with torch.no_grad():
    #         while batch_size * (i + 1) <= c:
    #             a = self.grid[batch_size * i:batch_size * (i + 1)]
    #             self.model(a.to(self.device))
    #             i += 1
    #         print('b---------- %d / %d' % (c, c))
    #     for hook in self.hooks_g:
    #         hook.remove()
    #     for layer in self.embeddings_g:
    #         self.et[layer] = torch.cat((self.et[layer], self.embeddings_g[layer].unsqueeze(0)), 0)
    #     self.i += 1

    def attn_demo(self, i):
        self.embeddings['attn'] = torch.Tensor([])

        def get_hook(name):
            def hook(model, input, output):
                self.embeddings[name] = torch.cat((self.embeddings[name], output.detach().cpu()), 0)

            return hook

        hook = self.name2decoder_module['attn'].register_forward_hook(get_hook('attn'))
        encoder_hidden = self.model.initHidden()
        max_length = 10
        encoder_outputs = torch.zeros(max_length, self.model.hidden_size, device=self.device)
        input_tensor = self.input[i]
        encoder_words = []
        for ei in range(len(input_tensor)):
            encoder_output, encoder_hidden = self.model(input_tensor[ei],
                                                        encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]
            encoder_words.append(self.info['switch'][input_tensor[ei].item()])

        decoder_input = torch.tensor([[0]], device=self.device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)
        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == 1:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(self.info['switch_'][topi.item()])

            decoder_input = topi.squeeze().detach()
        self.embeddings['attn'] = torch.softmax(self.embeddings['attn'], dim=1)
        self.embeddings['attn'] = self.embeddings['attn'][:, :len(input_tensor)]
        self.sens['input'] = encoder_words
        self.sens['output'] = decoded_words
        self.sens['target'] = self.targets[i]
        hook.remove()
        return self.embeddings['attn'], encoder_words, decoded_words, self.targets[i]

    def trans_demo(self, sen):
        self.embeddings['encoder.layers.0.self_attn.attn'] = torch.Tensor([])
        self.embeddings['encoder.layers.1.self_attn.attn'] = torch.Tensor([])
        start = self.embeddings['encoder.layers.0'].shape[1]
        data = self.info['data']
        greedy_decode = self.info['greedy_decode']
        model = self.model
        dev_en = [data.en_word_dict[w] for w in sen]
        en_sent = " ".join([data.en_index_dict[w] for w in dev_en])

        src = torch.from_numpy(np.array(dev_en)).long().to(self.device)
        src = src.unsqueeze(0)
        src_mask = (src != 0).unsqueeze(-2)

        out, values, indices = greedy_decode(model, src, src_mask, max_len=60, start_symbol=data.cn_word_dict["BOS"])

        indices = [[data.cn_index_dict[i] for i in ins] for ins in indices]
        translation = []
        for j in range(1, out.size(1)):
            sym = data.cn_index_dict[out[0, j].item()]
            if sym != 'EOS':
                translation.append(sym)
            else:
                break
        print("translation: %s" % " ".join(translation))
        embedding = torch.cat(
            (self.embeddings['encoder.layers.0.self_attn.attn'], self.embeddings['encoder.layers.1.self_attn.attn']), 0)
        end = self.embeddings['encoder.layers.0'].shape[1]

        return en_sent, translation, embedding, list(range(start, end)), values, indices

    def reset_embedding(self):
        for layer in self.hooks_1:
            self.embeddings_1[layer] = torch.Tensor([])
        # why(not) clear these(self.embeddings)? maybe not proper, need to summarize
        for layer in self.embeddings_g:
            self.embeddings_g[layer] = torch.Tensor([])
        for layer in self.connections:
            self.connections[layer] = torch.Tensor([])

    def activate_hooks(self):
        if len(self.hooks_list) == 0:
            for layer in self.hooks:
                self.hooks_list.append(self.name2module[layer].register_forward_hook(self.hooks[layer]))

    def deactivate_hooks(self):
        for hook in self.hooks_list:
            hook.remove()
        self.hooks_list.clear()

    def run(self, file_name='templates/index.html', batch_size=64, mode=0, foward_manual=True, auto_open=False, **info):
        if self.model:
            # 没有model的情况仅用于方便测试
            if not foward_manual:
                self.forward(batch_size, mode)
                if mode == 1:
                    self.attn_demo(0)
                if len(self.embeddings_g.keys()) != 0:
                    self.forward_b(batch_size)
            # remove hook in stage 0 and add hook in stage 1 to save storage.
            for hook in self.hooks_list:
                hook.remove()
            for layer in self.hooks_1:
                self.name2module[layer].register_forward_hook(self.hooks_1[layer])
        # ----------------------------------------
        dir_path = os.path.dirname(os.path.abspath(__file__))
        f = open(dir_path + '/' + file_name, 'w', encoding='utf-8')
        View.set_file(f)
        views = View.views
        vi = 0
        widgets = View.widgets
        # View.set_set_mode(1)
        for vc in self.view_components:
            data = pd.DataFrame()
            if vc.layer is None:
                if vc.view_class == CompositeView1:
                    data_s = []
                    for info in vc.view_info:
                        # for i in range(self.i):
                        for i in range(1):
                            data = pd.DataFrame()
                            # complement data for decision border
                            if info['element']['class'] == 'decision_border':
                                data['grid'] = self.grid.numpy().tolist()
                                data['idx'] = [i for i in range(self.grid.shape[0])]
                            # construct primitive data
                            for k in info['data']:
                                v = info['data'][k]
                                # convert str layer to data
                                if isinstance(v, str):
                                    if info['element']['class'] == 'decision_border':
                                        v = self.et[v][i]
                                    else:
                                        v = self.embeddings[v]
                                # construct id column
                                if 'idx' not in data and isinstance(v, (list, np.ndarray, torch.Tensor)):
                                    data['idx'] = [i for i in range(len(v))]
                                data[k] = df_tolist(v)
                            # compute transform in data
                            for tr in info['transform']:
                                if 'activation' in tr:
                                    data[tr['in']] = df_tolist(
                                        tr['activation'](torch.from_numpy(np.stack(data[tr['in']].values))))
                                if 'out' in tr:
                                    data[tr['out']] = df_tolist(tr['method'](np.stack(data[tr['in']].values)))
                            cm = plt.get_cmap('bwr')
                            data['color'] = list(
                                map(lambda x: colors.to_hex(cm(x), keep_alpha=True),
                                    data[info['element']['color']].values))
                            data_s.append(data)
                        # print(len(data_s), data_s[0].shape)
                    view = vc.view_class(vc.position, vc.size, data_s, [info['element'] for info in vc.view_info])
                elif vc.view_class == CompositeView:
                    data_s = []
                    for info in vc.view_info:
                        data = pd.DataFrame()
                        # complement data for decision border
                        if info['element']['class'] == 'decision_border':
                            data['grid'] = self.grid.numpy().tolist()
                            data['idx'] = [i for i in range(self.grid.shape[0])]
                        # construct primitive data
                        for k in info['data']:
                            v = info['data'][k]
                            # convert str layer to data
                            if isinstance(v, str):
                                if info['element']['class'] == 'decision_border':
                                    v = self.embeddings_g[v]
                                else:
                                    v = self.embeddings[v]
                            # construct id column
                            if 'idx' not in data and isinstance(v, (list, np.ndarray, torch.Tensor)):
                                data['idx'] = [i for i in range(len(v))]
                            data[k] = df_tolist(v)
                        # compute transform in data
                        for tr in info['transform']:
                            if 'activation' in tr:
                                data[tr['in']] = df_tolist(
                                    tr['activation'](torch.from_numpy(np.stack(data[tr['in']].values))))
                            if 'out' in tr:
                                data[tr['out']] = df_tolist(tr['method'](np.stack(data[tr['in']].values)))
                        cm = plt.get_cmap('bwr')
                        data['color'] = list(
                            map(lambda x: colors.to_hex(cm(x), keep_alpha=True), data[info['element']['color']].values))
                        data_s.append(data)
                    view = vc.view_class(vc.position, vc.size, data_s, [info['element'] for info in vc.view_info])
                    # # View.idm.register_id(1, data)
                elif vc.view_class == Gallery:
                    view = vc.view_class(vc.position, vc.size, reg_no=0, type=vc.view_info.get('type', 'img'),
                                         ref_pos=self.ref_pos, text=self.input, switch=self.info.get('switch'),
                                         nvt=self)
                elif vc.view_class == HeatMap:
                    cm = plt.get_cmap('Reds')
                    cm = vc.view_info.get('cm', plt.get_cmap(heat_map_config['cm']))
                    colors_ = list(map(lambda x: colors.to_hex(cm(x), keep_alpha=True), vc.view_info['values']))
                    data['color'] = colors_
                    view = vc.view_class(vc.position, vc.size, data, shape=vc.view_info['shape'],
                                         x_titles=vc.view_info.get('x_titles'), y_titles=vc.view_info.get('y_titles'))
                elif vc.view_class == ParallelCoordinate:
                    data['color'] = vc.view_info.get('colors', path_config['color'])
                    values = vc.view_info['values']
                    data['value'] = [[values[i][j] for j in range(len(values[0]))] for i in range(len(values))]
                    view = vc.view_class(vc.position, vc.size, data)
                elif vc.view_class == BarChart:
                    data['idx'] = [i for i in range(len(vc.view_info['values']))]
                    View.idm.register_id(1, data)
                    data['y'] = vc.view_info['values']
                    data['color'] = vc.view_info.get('colors', barchart_config['color'])
                    data['x'] = vc.view_info['titles']
                    view = vc.view_class(vc.position, vc.size, data, reg_no=1)
                elif vc.view_class == TextView:
                    view = vc.view_class(vc.position, vc.size, reg_no=1,
                                         plc=[v for v in views if isinstance(v, ParallelCoordinate1)][0])
                elif vc.view_class == SelectorView:
                    reg_ids = View.idm.get_reg_ids()
                    view = vc.view_class(vc.position, vc.size, reg_ids=reg_ids)
                elif issubclass(vc.view_class, Widget):
                    if vc.view_class == Input:
                        widget = vc.view_class(vc.position, vc.size,
                                               views=[self.named_views[name] for name in vc.view_info['views']])
                    elif vc.view_class == Select:
                        widget = vc.view_class(vc.position, vc.size,
                                               views=[self.named_views[name] for name in vc.view_info['views']],
                                               options=vc.view_info['options'])
            else:
                embedding = self.embeddings[vc.layer]
                # ----------------
                if vc.transform == 'TSNE':
                    position = TSNE().fit_transform(embedding)
                elif vc.transform is None:
                    position = embedding
                else:
                    position = vc.transform(embedding)
                # position: (size, dim) tensor or ndarray
                # ----------------
                if vc.view_class == ScatterPlot:
                    # cm
                    cm = vc.view_info.get('cm', circle_config['cm'])
                    if isinstance(cm, str):
                        cm = plt.get_cmap(cm)
                    # color_labels
                    color_labels = vc.view_info.get('color_labels', self.targets)
                    # colors
                    if 'colors' not in vc.view_info or vc.view_info['colors'] == 'default':
                        colors_ = circle_config['color']
                    elif vc.view_info['colors'] == 'labels':
                        colors_ = list(map(lambda x: colors.to_hex(cm(x), keep_alpha=True), color_labels))
                    else:
                        colors_ = vc.view_info['color']
                    data['idx'] = list(range(len(position)))
                    View.idm.register_id(0, data)
                    data['x'] = position[:, 0]
                    data['y'] = position[:, 1]
                    data['r'] = vc.view_info.get('point_size', circle_config['r'])
                    data['color'] = colors_
                    data['label'] = self.targets if vc.view_info.get('labels', 'default') == 'default' else \
                        vc.view_info['labels']
                    data['embedding'] = embedding.numpy().tolist()
                    data['opacity'] = circle_config['color']
                    view = vc.view_class(vc.position, vc.size, data, reg_no=0)
                elif vc.view_class == ParallelCoordinate1:
                    embedding1 = embedding.T
                    hid_selected = random.choices(list(range(embedding1.shape[0])), k=64)
                    # hid_selected = list(range(embedding1.shape[0]))
                    pos_begin = random.randint(0, embedding.shape[1])
                    n = 16
                    embedding1 = embedding1[hid_selected, pos_begin:pos_begin + n]
                    values = df_tolist(embedding1)
                    data['value'] = [[values[i][j] for j in range(len(values[0]))] for i in range(len(values))]
                    data['idx'] = [i for i in hid_selected]
                    data['color'] = vc.view_info.get('colors', path_config['color'])
                    View.idm.register_id(1, data)
                    x = 0
                    for i in range(1, len(self.ref_pos)):
                        if self.ref_pos[i] > pos_begin:
                            x = i - 1
                            y = pos_begin - self.ref_pos[x]
                            break
                    titles = []
                    for i in range(n):
                        titles.append([x, y])
                        y += 1
                        if self.ref_pos[x] + y == self.ref_pos[x + 1]:
                            x += 1
                            y = 0
                    titles = [self.info['switch'][self.input[titles[i][0]][titles[i][1]].item()] for i in
                              range(len(titles))]
                    # 下面的pos_begin参数只是用于调试
                    view = vc.view_class(vc.position, vc.size, data, reg_no=1, embedding=embedding, titles=titles,
                                         pos_begin=pos_begin, refs=0)
                elif vc.view_class == LinkMap:
                    View.idm.register_id(2)
                    view = vc.view_class(vc.position, vc.size, None, reg_no=2)
                elif vc.view_class == TransDemo:
                    # need generalization
                    view = vc.view_class(vc.position, vc.size, data=None, **vc.view_info, nvt=self,
                                         corpus_len=self.embeddings['encoder.layers.0'].shape[1],
                                         embeddings=self.embeddings)
                else:
                    view = vc.view_class(vc.position, vc.size, position)
            if issubclass(vc.view_class, Widget):
                widgets.append(widget)
            else:
                if vc.name is None:
                    name = 'view%d' % vi
                else:
                    name = vc.name
                vi += 1
                self.named_views[name] = view
                views.append(view)
        View.init_prev_r()
        xm = max([v.position[0] + v.size[0] for v in View.views]) + 1600
        ym = max([v.position[1] + v.size[1] for v in View.views]) + 1600
        head(f, views=views, size=[xm, ym])
        for v in views:
            v.core()
        for w in widgets:
            w.core()
        # default brush model and other complement
        f.write(f"""
        //mode.click();
        empty_r={View.idm.empty_r()};
        filter2.selectedIndex = 0;
        filter2_.dispatch('change');""")
        f.write("\n</script>")
        f.close()
        launch(auto_open)
