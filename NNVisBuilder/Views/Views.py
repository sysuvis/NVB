import pandas as pd
import json
import inspect
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import torch
# from NNVisBuilder.utils import MSelector
from NNVisBuilder.GlobalVariables import *
from NNVisBuilder.Data import Data, Filter
from NNVisBuilder.Views import View


# add record click position base on params after
class ScatterPlot(View):
    def __init__(self, data, position=[100, 100], size=None, highlighter=None, title=None, point_size=circle_config['r'],
                 colors=None, color_labels=None, cm=circle_config['cm1'], opacity=circle_config['opacity'], border_color='black',
                 click_view=False, highlight_border=False, save_range=False):
        """
        :param info: reg_no
        """
        super(ScatterPlot, self).__init__(data, position, size, highlighter, title=title, stroke_color=border_color)
        if point_size is not None:
            if not isinstance(point_size, Data):
                point_size = Data(point_size)
            point_size.views.append(self)
        self.point_size = point_size
        self.cm = cm
        self.opacity = opacity
        self.click_view = click_view
        self.highlight_border = highlight_border
        if colors is not None:
            if not isinstance(colors, Data):
                colors = Data(colors)
            colors.views.append(self)
        self.colors = colors
        if color_labels is not None:
            if not isinstance(color_labels, Data):
                color_labels = Data(color_labels)
            color_labels.views.append(self)
        self.color_labels = color_labels
        if highlighter is not None and highlighter.core() is None:
            highlighter.set_style(circle_config['style'])
        self.xs = None
        self.ys = None
        self.save_range = save_range

    def generate_vis_data(self):
        if isinstance(self.data, Data):
            value = self.data.value_().copy()
        else:
            value = self.data.copy()
        if len(value) == 0:
            return json.dumps([])
        if not self.save_range or self.xs is None:
            self.xs = [np.min(value[:, 0]), np.max(value[:, 0])]
            self.ys = [np.min(value[:, 1]), np.max(value[:, 1])]
        x0, x1 = self.xs
        y0, y1 = self.ys
        value[:, 0] = (0.01 + 0.98 * ((value[:, 0] - x0) / (x1 - x0))) * self.size[0]
        value[:, 1] = self.size[1] - (0.01 + 0.98 * ((value[:, 1] - y0) / (y1 - y0))) * self.size[1]
        data = pd.DataFrame()
        data['cx'] = value[:, 0]
        data['cy'] = value[:, 1]
        colors_ = None
        if self.color_labels is None:
            if isinstance(self.colors, Data):
                colors_ = self.colors.value_()
            elif self.colors is None:
                colors_ = circle_config['color']
            else:
                colors_ = self.colors
        else:
            cm = plt.get_cmap(self.cm) if isinstance(self.cm, str) else self.cm
            if isinstance(self.color_labels, Data):
                colors_ = self.color_labels.value_()
            else:
                colors_ = self.color_labels
            if isinstance(colors_, torch.Tensor):
                colors_ = colors_.cpu().detach().numpy()
            if isinstance(colors_, np.ndarray):
                colors_ = colors_.reshape(-1).tolist()
            colors_ = list(map(lambda x: colors.to_hex(cm(x), keep_alpha=True), colors_))
        data['color'] = colors_
        data['r'] = self.point_size.value_()
        data['idx'] = list(range(data.shape[0]))
        return data.to_json(orient='records')

    def core(self):
        super(ScatterPlot, self).core()
        View.f.write(f"""
brush{self.idx}.on("end", e => {{
        if(e.selection){{
            const [[x0, y0], [x1, y1]] = e.selection;
            brush_ids = [];
            g{self.idx}.selectAll("circle")
                .each(d => {{
                    const cx = d.cx, cy = d.cy, pos = brush_ids.indexOf(d.idx);
                    if (cx >= x0 && cx <= x1 && cy >= y0 && cy <= y1) {{
                        if(pos == -1) brush_ids.push(d.idx);
                    }}
                    else if(pos != -1) brush_ids.splice(pos, 1);
                }});  
            d3.json(`/brush/{self.idx}?value=${{brush_ids}}`)
                .then(r => {{
                    for(let i of r[0]) triggers[i].dispatch('click');
                    for(let i of r[1]) highlighters[i].dispatch('click');
                }});
        }}
        else{{
            brush_ids = [];
            d3.json(`/brush/{self.idx}?value=${{brush_ids}}`)
                .then(r => {{
                    for(let i of r[0]) triggers[i].dispatch('click');
                    for(let i of r[1]) highlighters[i].dispatch('click');
                }});
        }}
    }});
trigger{self.idx}.on('click', e => {{
    d3.json('/trigger/{self.idx}').then(r => {{
        g{self.idx}.selectAll('circle').remove();
        g{self.idx}.selectAll('circle')
            .data(r).enter().append('circle')
            .attr('r', d => d.r)
            .attr('cx', d => d.cx)
            .attr('cy', d => d.cy) 
            .attr('fill', d => d.color)
            .attr('opacity', {self.opacity})
            //.attr('opacity', d => d.opacity)""" + (f"""
            .on('click', e => {{
                const idx = {select_this}.datum().idx;
                d3.json(`/click/{self.idx}?value=${{idx}}`).then(r => {{
            """ + (f"""
                    d3.selectAll('rect.border_').attr('stroke-width', 1);
                    g{self.idx}.select('rect.border_').attr('stroke-width', '3');
            """ if self.highlight_border else '') + f"""
                    for(let i of r[0]) triggers[i].dispatch('click');
                    for(let i of r[1]) highlighters[i].dispatch('click');
                }});
            }})
            """ if not self.click_view else ".attr('pointer-events', 'none')") + f""";
    }});
}});
trigger{self.idx}.dispatch('click');
highlighter{self.idx}.on('click', e => {{
    d3.json('/highlighter/{self.idx}').then(r => {{
        g{self.idx}.selectAll('circle'){self.highlighter.core() if self.highlighter is not None else ';'}
    }});
}});
        """.strip() + "\n")
        if self.click_view:
            View.f.write(f"""
g{self.idx}.select('rect.border_')
    .attr('opacity', 0.4)
    .on('click', e => {{
        d3.selectAll('rect.border_').attr('stroke-width', 1);
        g{self.idx}.select('rect.border_').attr('stroke-width', '3');
        d3.json(`/click/{self.idx}?value=-1`).then(r => {{
            for(let i of r[0]) triggers[i].dispatch('click');
            for(let i of r[1]) highlighters[i].dispatch('click');
        }});
    }});
            """)

    def click(self, request_args):
        value = int(request_args.get('value'))
        self.last_value = value
        View.update_list.clear()
        View.highlight_list.clear()
        if self.click_ is not None:
            if len(inspect.signature(self.click_).parameters) == 2:
                self.click_(value, [self.position[0]+self.size[0]/2, self.position[1]+self.size[1]/2])
            else:
                self.click_(value)
        return json.dumps([View.update_list, View.highlight_list])

    def brush(self, request_args):
        value = request_args.get('value')
        if value == '':
            value = []
        else:
            value = value.split(',')
        value = [int(x) for x in value]
        View.update_list.clear()
        View.highlight_list.clear()
        self.brush_(value)
        return json.dumps([View.update_list, View.highlight_list])


class LineChart(View):
    def __init__(self, data, position=[100, 100], size=None, highlighter=None, title=None, colors=None):
        super(LineChart, self).__init__(data, position, size, highlighter, title=title)
        self.colors = colors

    def generate_vis_data(self):
        pass


class Tooltip1(View):
    def __init__(self, data, position=[100, 100], size=None):
        super(Tooltip1, self).__init__(data, position, size, border=False)

    def generate_vis_data(self):
        return json.dumps(self.data.value_().tolist())

    def core(self):
        super(Tooltip1, self).core()
        View.f.write(f"""
trigger{self.idx}.on('click', e => {{
    d3.json('/trigger/{self.idx}').then(r => {{
        console.log(x_click, y_click, 'fff');
        toolTip.attr('style', 'left:' + x_click + 'px' + ';top:' + y_click + 'px').selectAll('*').remove();
        toolTip.style('height', '70px').style('width', '120px').on('click', e => {{
            // toolTip.classed('hidden', true);
        }});
        toolTip.selectAll('input')
            .data(r).enter().append('input')
            .attr('value', d => d.toFixed(2))
            .style('width', '112px')
            .attr('id', (d, i) => `input{self.idx}_${{i+1}}`);
        toolTip.append('button')
            .text('update')
            .style('position', 'absolute')
            .style('top', '50px')
            .style('left', '5px')
            .style('width', '60px')
            .on('click', e => {{
                toolTip.classed('hidden', true);
                const v1 = document.getElementById('input{self.idx}_1').value, v2 = document.getElementById('input{self.idx}_2').value;
                d3.json(`/click/{self.idx}?v1=${{v1}}&v2=${{v2}}`).then(r => {{
                    for(let i of r[0]) triggers[i].dispatch('click');
                    for(let i of r[1]) highlighters[i].dispatch('click');
                }});
            }});
        toolTip.append('button')
            .text('cancel')
            .style('position', 'absolute')
            .style('top', '50px')
            .style('left', '65px')
            .style('width', '60px')
            .on('click', e => {{
                toolTip.classed('hidden', true);
            }});
        toolTip.classed('hidden', false);
    }});
}});
        """)

    def click(self, request_args):
        v1 = float(request_args.get('v1'))
        v2 = float(request_args.get('v2'))
        # special no need add update_list
        self.data.update([v1, v2])
        View.update_list.clear()
        View.highlight_list.clear()
        self.click_(v1, v2)
        print(View.update_list)
        return json.dumps([View.update_list, View.highlight_list])


class Tooltip(View):
    def __init__(self, data, position=[100, 100], size=None, prefix='', suffix='png'):
        super(Tooltip, self).__init__(data, position, size, border=False)
        self.prefix = prefix
        if isinstance(self.prefix, Data):
            prefix.views.append(self)
        self.suffix = suffix

    def generate_vis_data(self):
        r = {'position': self.position}
        value = self.data.tolist()
        img = [self.prefix.value_() + '-' + str(x) + '.' + self.suffix for x in value]
        r['img'] = img
        return json.dumps(r)

    def core(self):
        super(Tooltip, self).core()
        View.f.write(f"""
trigger{self.idx}.on('click', e => {{
    d3.json('/trigger/{self.idx}').then(r => {{
        toolTip1.attr('style', 'left:' + r['position'][0] + 'px' + ';top:' + r['position'][1] + 'px').selectAll('*').remove();
        toolTip1.style('height', '88px').style('width', '200px').on('click', e => {{
            toolTip1.classed('hidden', true);
        }});
        toolTip1.selectAll('img')
            .data(r['img'])
            .enter()
            .append('img')
            .style('width', '20%')
            .attr('src', d => `/static/img/${{d}}`);
        toolTip1.classed('hidden', false);
    }});
}});
        """)


class Picture(View):
    def __init__(self, data, position, size=None, title=None, prefix='', suffix='png'):
        super(Picture, self).__init__(data, position, size, title=title)
        self.prefix = prefix
        self.suffix = suffix

    def generate_vis_data(self):
        r = str(self.data.value_())
        if self.prefix != '':
            r = self.prefix + '-' + r + '.' + self.suffix
        return json.dumps(r)

    def core(self):
        super(Picture, self).core()
        View.f.write(f"""
const div{self.idx} = d2.append('div').classed('div4', true).style('position', 'absolute');
div{self.idx}.style('width', '{self.size[0]}px')
    .style('height', '{self.size[1]}px')
    .style('left', '{self.position[0]}px')
    .style('top', '{self.position[1]}px');
trigger{self.idx}.on('click', e => {{
    d3.json('/trigger/{self.idx}').then(r => {{
        div{self.idx}.select('img').remove();
        div{self.idx}.append('img')
            .attr('src', d => `/static/img/${{r}}`)
            .style('width', '100%')
            .style('height', '100%')
            .on('click', e => {{
                d3.json(`/click/{self.idx}?value=-1`).then(r => {{
                    for(let i of r[0]) triggers[i].dispatch('click');
                    for(let i of r[1]) highlighters[i].dispatch('click');
                }});
            }});
    }});
}});
trigger{self.idx}.dispatch('click');
        """)

    def click(self, request_args):
        View.update_list.clear()
        View.highlight_list.clear()
        if self.click_ is not None:
            if len(inspect.signature(self.click_).parameters) == 2:
                self.click_(self.prefix, [self.position[0] + self.size[0] * 1.8 / 2, self.position[1] + self.size[1] * 1.8 / 2])
            else:
                self.click_(self.prefix)
        return json.dumps([View.update_list, View.highlight_list])


class SentenceList(View):
    def __init__(self, data, position, size=None, title=None, cell_size=None):
        """
        :param data: Vector
        """
        super(SentenceList, self).__init__(data, position, size, title=title, border=False)
        self.cell_size = cell_size
        self.shape = None

    def generate_vis_data(self):
        data = pd.DataFrame()
        value = self.data.value_().reshape(-1)
        data['text'] = value
        self.shape = self.data.size()
        if self.cell_size is None:
            if self.size is not None:
                width = self.size[0] / self.shape[1]
                height = self.size[1] / self.shape[0]
            else:
                width, height = 25, 15
        else:
            width, height = self.cell_size
        data['font_size'] = [width * 0.3 if len(w) < 7 else 5 / len(w) * width * 0.4 for w in value]
        data['width'] = width
        data['height'] = height
        data['idx'] = list(range(data.shape[0]))
        data['x'] = [width * j for i in range(self.shape[0]) for j in range(self.shape[1])]
        data['y'] = [height * i for i in range(self.shape[0]) for j in range(self.shape[1])]
        data['color'] = 'black'
        return data.to_json(orient='records')

    def core(self):
        super(SentenceList, self).core()
        View.f.write(f"""
trigger{self.idx}.on('click', e => {{
    d3.json('/trigger/{self.idx}').then(r => {{
        g{self.idx}.selectAll('text').remove();
        g{self.idx}.selectAll('text')
            .data(r).enter().append('text')
            .attr('fill', 'black')
            .attr('x', d => d.x + d.width / 2)
            .attr('y', d => d.y + d.height / 2)
            .attr('dy', '.35em')
            .attr('text-anchor', 'middle')
            .attr('font-size', d => d.font_size)
            .text(d => d.text)
            .on('click', e => {{
            d3.json(`/click/{self.idx}?value=${{{select_this}.datum().idx}}`).then(r => {{
                    for(let i of r[0]) triggers[i].dispatch('click');
                    for(let i of r[1]) highlighters[i].dispatch('click');
                }});
            }});
    }});
}});
        """)


class Gallery(View):
    def __init__(self, data, position, size=None, title=None, suffix='png'):
        super(Gallery, self).__init__(data, position, size, title=title, border=False)
        self.suffix = suffix

    def generate_vis_data(self):
        r = self.data.tolist()
        r = ['%d.%s' % (x, self.suffix) for x in r]
        return json.dumps(r)

    def core(self, views=None):
        super(Gallery, self).core()
        View.f.write(f"""
const div{self.idx} = d2.append('div').classed('div3', true);
div{self.idx}.style('width', '{self.size[0]}px')
    .style('height', '{self.size[1]}px')
    .style('left', '{self.position[0]}px')
    .style('top', '{self.position[1]}px');
trigger{self.idx}.on('click', e => {{
    d3.json('/trigger/{self.idx}').then(r => {{
        div{self.idx}.selectAll('img').remove();
        div{self.idx}.selectAll('img')
            .data(r)
            .enter()
            .append('img')
            .attr('src', d => `/static/img/${{d}}`)
            .style('width', '25%');
    }});
}});
        """)


#     def response(self, views=None, back=False):
#         if self.info.get('type', 'img') == 'img':
#             if not back:
#                 return f"""
#     g{self.idx}_.selectAll('img').remove();
#     g{self.idx}_.selectAll('img')
#         .data(r[{self.reg_no}])
#         .enter()
#         .append('img')
#         .attr('src', d => `/static/img/${{d}}.png`)
#         .style('width', "25%")
#         .on('click', e => {{
#             selected_back = [{select_this}.datum()];
#             {newline.join([view.response(views, back=True) for view in views if isinstance(view, (ScatterPlot, LinkMap))])}
#         }});
#                 """.strip() + "\n"
#             else:
#                 return ''
#         elif self.info.get('type', 'img') == 'text':
#             return f"""
# //save the r for select_back
# const r_ = r;
# if(r != false){{
# d3.json(`/ite/?ids=${{{'selected_back' if back else 'r'}[{self.reg_no}]}}&mc=${{{'selected_back' if back else 'r'}[100]}}&l=${{{'selected_back' if back else 'r'}[101]}}`).then(r => {{
#             g{self.idx}.selectAll('text').remove();
#             g{self.idx}.selectAll('text')
#                 .data(r['text'])
#                 .enter()
#                 .append('text')
#                 .text(d => d.text)
#                 .attr('x', d => `${{d.x}}px`)
#                 .attr('y', d => `${{d.y}}px`)
#                 .attr('fill', d => d.color)
#                 .attr('dx', 0)
#                 .attr('dy', 0)
#                 .on('click', e => {{
#                     const d = {select_this}.datum();
#                     d3.json(`/gtl/te?s=${{d.s}}&i_=${{d.i_}}`).then(selected_back => {{
#                         {newline.join([view.response(views, back=True) for view in views if isinstance(view, (ScatterPlot, LinkMap))])}
#                     }});
#                 }});
#             g{self.idx}.selectAll("rect.r{self.idx}").remove();
#             if(typeof(r['mc']) != 'undefined'){{
#                 g{self.idx}.selectAll("rect.r{self.idx}")
#                     .data(r['mc'])
#                     .enter()
#                     .append("rect")
#                     .classed('r{self.idx}', true)
#                     .attr('x', d => d.x)
#                     .attr('y', d => d.y)
#                     .attr('width', d => d.width)
#                     .attr('height', d => d.height)
#                     .attr('fill', d => d.color)
#                     .on('mouseover', e => {{
#                         const d_ = {select_this}.datum();
#                         let y_pos = e.clientY + document.documentElement.scrollTop
#                         let x_pos = e.clientX + document.documentElement.scrollLeft
#                         toolTip.attr("style", "left:" + x_pos + "px" + ";top:" + y_pos + "px").selectAll('*').remove();
#                         toolTip.append('p').text(d_.mc);
#                         toolTip.classed("hidden", false);
#                         g{self.idx}.selectAll('text')
#                             .attr('fill', d => {{
#                                 if(d.idx == d_.idx) return 'red';
#                                 else return d.color;
#                             }});
#                     }})
#                     .on('mouseout', e => {{
#                         toolTip.classed("hidden", true);
#                         g{self.idx}.selectAll('text').attr('fill', d => d.color);
#                     }});
#             }}
#         }});
#         }}
#             """.strip() + "\n"

# def ite(self, info):
#     """
#     transform the idx to text
#     :param info:
#     :return:
#     """
#     r = {}
#     ids = info.get('ids')
#     if ids == '':
#         r['text'] = []
#         return json.dumps(r)
#     ids = [int(x) for x in ids.split(',')]
#     data = pd.DataFrame()
#     # s: sentence index, i: word index in the sentence, i_: origin index
#     x = []
#     y = []
#     t = []
#     s = []
#     w = []
#     i_ = []
#     color = []
#     text = torch.cat(self.info['text'])
#     for row, idx in enumerate(ids):
#         b = 0
#         for i in range(10):
#             # 如果偶然选到了最后的idx，是会报错的，但是这里先不处理
#             if idx + i >= len(text):
#                 continue
#             a = self.info['switch'][text[idx + i].item()]
#             t.append(a)
#             color.append('black')
#             x.append(b)
#             b += len(a) * 10 + 10
#             y.append(row * 30)
#             pos = 0
#             for j in range(1, len(self.info['ref_pos'])):
#                 if self.info['ref_pos'][j] > idx + i:
#                     pos = j - 1
#                     break
#             s.append(pos)
#             w.append(i)
#             i_.append(idx)
#         # b = 0
#         # for i, v in enumerate(self.info['text'][pos]):
#         #     t.append(self.info['switch'][v[0].item()])
#         #     color.append('red' if i == idx - self.info['ref_pos'][pos] else 'black')
#         #     x.append(b)
#         #     b += len(self.info['switch'][v[0].item()]) * 10 + 10
#         #     y.append(row * 30)
#         #     s.append(pos)
#         #     w.append(i)
#         #     i_.append(idx)
#     data['x'] = x
#     data['y'] = y
#     data['color'] = color
#     data['text'] = t
#     data['s'] = s
#     data['w'] = w
#     data['i_'] = i_
#     data['font_size'] = 10
#     data['idx'] = list(range(data.shape[0]))
#     r['text'] = data.to_dict(orient='records')
#     mc = info.get('mc')
#     if mc != 'undefined':
#         data = pd.DataFrame()
#         shape = [10, 10]
#         size = [400, 300]
#         mc = [int(x) for x in mc.split(',')]
#         if idx + i >= len(mc):
#             r['mc'] = {}
#         else:
#             mc = [mc[idx + i] for idx in ids for i in range(10) if idx + i < len(mc)]
#             values = [x / int(info.get('l')) for x in mc]
#             cm = plt.get_cmap('Blues')
#             colors_ = list(map(lambda x: colors.to_hex(cm(x), keep_alpha=True), values))
#             data['mc'] = mc
#             data['color'] = colors_
#             w = size[0] / shape[0]
#             h = size[1] / shape[1]
#             data['width'] = w
#             data['height'] = h
#             data['x'] = [(i % shape[0]) * w + 600 for i in range(data.shape[0])]
#             data['y'] = [(i // shape[0]) * h - 20 for i in range(data.shape[0])]
#             data['idx'] = list(range(data.shape[0]))
#             r['mc'] = data.to_dict(orient='records')
#     return json.dumps(r)

# def gtl(self, info):
#     # gallery to linkmap
#     r = {}
#     s = int(info.get('s'))
#     i_ = int(info.get('i_'))
#     for reg_no in View.idm.get_all_reg_no():
#         if reg_no == 2:
#             r[2] = {}
#             data = pd.DataFrame()
#             embedding, input_words, output_words, target = self.info['nvt'].attn_demo(s)
#             target = target.split(' ')
#             data['attn'] = embedding.reshape(-1)
#             data['idx'] = list(range(data.shape[0]))
#             data['width'] = data['attn'] * 8
#             if 'color' not in data:
#                 data['color'] = 'grey'
#             ends = []
#             for i in range(len(output_words)):
#                 for j in range(len(input_words)):
#                     ends.append({'source': [200 * i + 50, 30], 'target': [200 * j + 50, 330]})
#             data['ends'] = ends
#             r[2]['element'] = data.to_dict('records')
#             w_data = pd.DataFrame()
#             w_data['x'] = [200 * j for j in range(len(input_words))] + [200 * i for i in
#                                                                         range(len(output_words))] + \
#                           [200 * i for i in range(len(target))]
#             w_data['y'] = [350 for j in range(len(input_words))] + [20 for i in
#                                                                     range(len(output_words))] + \
#                           [0 for i in range(len(target))]
#             w_data['text'] = input_words + output_words + target
#             w_data['idx'] = [-1 for j in range(len(input_words))] + [i for i in
#                                                                      range(len(output_words))] + \
#                             [-1 for i in range(len(target))]
#             r[2]['view'] = w_data.to_dict('records')
#         elif reg_no == 0:
#             r[0] = [i_]
#         elif reg_no not in r:
#             r[reg_no] = []
#     r['il'] = len(input_words)
#     r['ol'] = len(output_words)
#     return json.dumps(r)


class ParallelCoordinate(View):
    def __init__(self, data, position=[100, 100], size=[1250, 300], highlighter=None, title=None, x_titles=None,
                 threshold=None, selector='on_threshold', colors=None, legend=None):
        """
        :param data: matrix type
        :param threshold: default set to None to enable no threshold, for demonstration we set it to 0.
        """
        super(ParallelCoordinate, self).__init__(data, position, size, highlighter, title=title, border=False)
        selectors = {'default': f"""
        for(let i=0;i<temp.length;i++){{
            if(sx{self.idx}(temp[i][0])>=x0 && sx{self.idx}(temp[i][0])<=x1){{
                flag = true;
                if(sy{self.idx}(temp[i][1])>y1 || sy{self.idx}(temp[i][1])<y0){{
                    flag1 = false;
                    break;
                }}
            }}
        }}
                    """,
                     'on_threshold': f"""
        for(let i=0;i<temp.length;i++){{
            if(sx{self.idx}(temp[i][0])>=x0 && sx{self.idx}(temp[i][0])<=x1){{
                flag = true;
                if(temp[i][1]<threshold{self.idx}){{
                    flag1 = false;
                    break;
                }}
            }}
        }}
                    """
                     }
        self.selector = selectors[selector]
        # self.highlighters = []
        # if highlighter is not None and not isinstance(highlighter, list):
        #     self.highlighters = [highlighter]
        # for highlighter in self.highlighters:
        #     if highlighter.core() is None:
        #         highlighter.set_style(path_config['style'])
        if highlighter is not None and highlighter.core() is None:
            highlighter.set_style(path_config['style'])
        self.click_ = lambda value: self.highlighter.update(value)
        self.colors = colors
        if x_titles is not None:
            if not isinstance(x_titles, Data):
                x_titles = Data(x_titles)
            x_titles.views.append(self)
        self.titles = x_titles
        # if '-' in str in titles, the axis display will go wrong but no influence.
        if threshold is not None:
            if not isinstance(threshold, Data):
                threshold = Data(threshold)
            threshold.views.append(self)
        self.threshold = threshold
        self.legend = legend

    def generate_vis_data(self):
        r = {}
        data = pd.DataFrame()
        if self.titles is None:
            titles_ = list(range(self.data.size()[1]))
        else:
            titles_ = ['%d-%s' % (i, title) for i, title in enumerate(self.titles.value_())]
        data['path'] = [[[titles_[i], float(x[i])] for i in range(len(x))] for x in self.data.value_()]
        data['color'] = path_config['color'] if self.colors is None else self.colors
        data['idx'] = list(range(data.shape[0]))
        r['path'] = data.to_dict(orient='records')
        r['titles'] = titles_
        y_min = np.min(self.data.value_())
        y_max = np.max(self.data.value_())
        r['y'] = [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)]
        if self.threshold is not None:
            t = {'path': [[titles_[0], self.threshold.value_()], [titles_[-1], self.threshold.value_()]],
                 'color': '#E25440', 'value': self.threshold.value_()}
            r['threshold'] = [t]
        else:
            r['threshold'] = []
        return json.dumps(r)

    def core(self):
        super(ParallelCoordinate, self).core()
        View.f.write(f"""
// scale can be defined in response.
const sx{self.idx} = d3.scaleBand().paddingInner(1).range([0, rw{self.idx}]);
const sy{self.idx} = d3.scaleLinear().range([rh{self.idx}, 0]);
let threshold{self.idx} = 0;
brush{self.idx}.on("end", e => {{
        if(e.selection){{
            const [[x0, y0], [x1, y1]] = e.selection;
            brush_ids.splice(0, brush_ids.length);
            g{self.idx}.selectAll('path.a')
                .each(d => {{
                    const temp = d.path;
                    let flag = false, flag1 = true;
                    {self.selector}
                    pos = brush_ids.indexOf(d.idx);
                    if(flag && flag1){{
                        if(pos == -1) brush_ids.push(d.idx);
                    }}
                    else{{
                        if(pos != -1) brush_ids.splice(pos, 1);
                    }}
                }});
            d3.json(`/brush/{self.idx}?value=${{brush_ids}}&y0=${{sy{self.idx}.invert(y0)}}&y1=${{sy{self.idx}.invert(y1)}}&threshold=${{threshold{self.idx}}}`)
                .then(r => {{
                    for(let i of r[0]) triggers[i].dispatch('click');
                    for(let i of r[1]) highlighters[i].dispatch('click');
                }});
        }}
        else{{
            brush_ids = [];
            d3.json(`/brush/{self.idx}?value=${{brush_ids}}&y0=${{sy{self.idx}.invert(y0)}}&y1=${{sy{self.idx}.invert(y1)}}&threshold=${{threshold{self.idx}}}`)
                .then(r => {{
                    for(let i of r[0]) triggers[i].dispatch('click');
                    for(let i of r[1]) highlighters[i].dispatch('click');
                }});
        }}
    }});
trigger{self.idx}.on('click', () => {{
    d3.json('/trigger/{self.idx}').then(r => {{
        sx{self.idx}.domain(r['titles']);
        sy{self.idx}.domain(r['y']);
        {self.axis()}
        g{self.idx}.selectAll('path.a').remove();
        const line{self.idx} = d3.line();
        g{self.idx}.selectAll('path.a')
            .data(r['path']).enter().append('path')
            .classed('a', true)
            .attr('d', d => line{self.idx}(d.path.map(v => [sx{self.idx}(v[0]), sy{self.idx}(v[1])])))
            .attr('stroke', d => d.color)
            .attr('fill', 'none')
            .attr('opacity', 1);
        if(r['threshold'].length != 0) threshold{self.idx} = r['threshold'][0]['value'];
        g{self.idx}.selectAll('path.b').remove();
        g{self.idx}.selectAll('path.b')
            .data(r['threshold']).enter().append('path')
            .classed('b', true)
            .attr('d', d => line{self.idx}(d.path.map(v => [sx{self.idx}(v[0]), sy{self.idx}(v[1])])))
            .attr('stroke', d => d.color)
            .attr('fill', 'none')
            .style("stroke-dasharray", "3, 3")
            .attr('opacity', 1.0);
        {self.bind_brush()}
    }});
}});
trigger{self.idx}.dispatch('click');
highlighter{self.idx}.on('click', e => {{
    d3.json('/highlighter/{self.idx}').then(r => {{
        g{self.idx}.selectAll('path.a'){self.highlighter.core() if self.highlighter is not None else ';'}
    }});
}});""" + (f"""
var dl = {self.legend};
var gl = g{self.idx}.append('g').attr('transform', `translate(${{rw{self.idx}-130}}, ${{rh{self.idx}-110}})`)
gl.append('rect')
    .classed('c', true)
    .attr('x', 0).attr('y', 0)
    .attr('opacity', 0.1)
    .attr('width', 134).attr('height', 110)
    .attr("fill", 'gray');
gl.selectAll('path.c')
    .data(dl).enter().append('path')
    .classed('c', true)
    .attr('d', (d, i) => d3.line()([[5, 10+30*i], [50, 10+30*i]]))
    .attr('stroke', d => d.color);
gl.selectAll('text.c')
    .data(dl).enter().append('text')
    .classed('c', true)
    .attr('fill', 'black')
    .attr('x', d => 55)
    .attr('y', (d, i) => 10+30*i)
    .attr('dy', '.35em')
    //.attr('text-anchor', 'middle')
    .text(d => d.label);
        """ if self.legend is not None else ''))

    def brush(self, request_args):
        value = request_args.get('value')
        if value == '':
            value = []
        else:
            value = value.split(',')
        y0 = float(request_args.get('y0', 0))
        y1 = float(request_args.get('y1', 0))
        View.info['y0'] = y0
        View.info['y1'] = y1
        View.info['threshold'] = float(request_args.get('threshold', 0))
        value = [int(x) for x in value]
        View.update_list.clear()
        View.highlight_list.clear()
        if len(inspect.signature(self.brush_).parameters) == 1:
            self.brush_(value)
        elif len(inspect.signature(self.brush_).parameters) == 2:
            self.brush_(value, self.threshold)
        elif len(inspect.signature(self.brush_).parameters) == 3:
            self.brush_(value, y0, y1)
        else:
            print("Wrong brush handler.")
        return json.dumps([View.update_list, View.highlight_list])

    def axis(self):
        return f"""
const ax{self.idx} = d3.axisBottom(sx{self.idx});
const ay{self.idx} = d3.axisLeft(sy{self.idx});
if(typeof(r['titles'][0]) == 'number'){{
    const tv{self.idx} = r['titles'].filter((d, i) => i % Math.ceil(r['titles'].length/10) == 0);
    if((r['titles'].length-1) % Math.ceil(r['titles'].length/20) != 0) tv{self.idx}.push(r['titles'][r['titles'].length-1]);
    ax{self.idx}.tickValues(tv{self.idx});
}}
else{{
    ax{self.idx}.tickFormat(d => d.slice(d.indexOf('-')+1));
}}

g{self.idx}.append('g').call(ax{self.idx}).attr('transform', 'translate(0, {self.size[1]})').attr('fill', 'none');
g{self.idx}.append('g').call(ay{self.idx}).attr('fill', 'none');
        """.strip() + "\n"


class BarChart(View):
    def __init__(self, data, position, size, highlighter=None, titles=None, show_titles=None, max_value=None, padding=0.2, title=None):
        """
        info: max_height, reg_no
        """
        if not isinstance(data, Data):
            data = Data(data)
        super(BarChart, self).__init__(data, position, size, highlighter, title=title, border=False)
        if highlighter is not None and highlighter.core() is None:
            highlighter.set_style(barchart_config['bc_red'])
        self.max_value = max_value
        if titles is not None:
            if not isinstance(titles, Data):
                titles = Data(titles)
            titles.views.append(self)
        self.titles = titles
        self.show_titles = show_titles
        self.padding = padding

    def generate_vis_data(self):
        r = {}
        data = pd.DataFrame()
        if self.max_value is None:
            max_value = self.data.value_().max() * 20 / 19
        else:
            max_value = self.max_value
        data['height'] = 0.95 * self.size[1] / max_value * self.data.value_()
        if self.titles is None:
            r['titles'] = list(range(data.shape[0]))
        else:
            r['titles'] = self.titles.value_().tolist()
        data['x'] = r['titles']
        data['color'] = 'blue'
        data['idx'] = list(range(data.shape[0]))
        r['rect'] = data.to_dict(orient='records')
        r['max_value'] = float(max_value)
        return json.dumps(r)

    def core(self):
        super(BarChart, self).core()
        View.f.write(f"""
const sx{self.idx} = d3.scaleBand()
    .range([0, rw{self.idx}]).padding({self.padding});
const sy{self.idx} = d3.scaleLinear()
    .range([rh{self.idx}, 0]);
trigger{self.idx}.on('click', () => {{
    d3.json('/trigger/{self.idx}').then(r => {{
        sx{self.idx}.domain(r['titles']);
        sy{self.idx}.domain([0, r['max_value']]);
        {self.axis()}
        g{self.idx}.selectAll('rect.a').remove();
        g{self.idx}.selectAll('rect.a')
            .data(r['rect']).enter().append('rect')
            .classed('a', true)
            .attr('width', d => sx{self.idx}.bandwidth())
            .attr('height', d => d.height)
            .attr('x', d => sx{self.idx}(d.x))
            .attr('y', d => {self.size[1]} - d.height)
            .attr('stroke', d => d.color)
            .attr('fill', d => d.color)
            .on('click', e => {{
                // all need to record position
                y_click = e.clientY + document.documentElement.scrollTop;
                x_click = e.clientX + document.documentElement.scrollLeft;
                d3.json(`/click/{self.idx}?value=${{{select_this}.datum().idx}}`).then(r => {{
                    for(let i of r[0]) triggers[i].dispatch('click');
                    for(let i of r[1]) highlighters[i].dispatch('click');
                }});
            }});
    }});
}});
trigger{self.idx}.dispatch('click');
highlighter{self.idx}.on('click', e => {{
    d3.json('/highlighter/{self.idx}').then(r => {{
        g{self.idx}.selectAll('rect.a'){self.highlighter.core() if self.highlighter is not None else ';'}
    }});
}});
        """.strip() + "\n")

    def click(self, request_args):
        value = int(request_args.get('value'))
        self.last_value = value
        View.update_list.clear()
        View.highlight_list.clear()
        self.click_(value)
        return json.dumps([View.update_list, View.highlight_list])

    def axis(self):
        return f"""
const ax{self.idx} = d3.axisBottom(sx{self.idx}).tickSizeOuter(0), ay{self.idx} = d3.axisLeft(sy{self.idx});
g{self.idx}.selectAll('g[class*=axis_]').remove();
g{self.idx}.append('g').call(ax{self.idx}).attr('transform', 'translate(0, {self.size[1]})').attr('fill', 'none')
.classed('axis_x', true)""" + (f".selectAll('.tick').remove()" if self.titles is None else '') + f""";
g{self.idx}.append('g').call(ay{self.idx}).attr('fill', 'none').classed('axis_y', true);
g{self.idx}.selectAll('.tick line').attr('stroke', '#d3d3d3');
g{self.idx}.select('.axis_x').selectAll('.tick line').attr('stroke', 'white');
g{self.idx}.selectAll('.tick text').attr('fill', '#708090');
        """.strip() + "\n"


class PointChart(View):
    def __init__(self, data, position, size, data1=None, highlighter=None, titles=None, show_titles=None, padding=0.2, title=None, e_idx=None):
        """
        info: max_height, reg_no
        """
        if not isinstance(data, Data):
            data = Data(data)
        super(PointChart, self).__init__(data, position, size, highlighter, title=title, border=False)
        if highlighter is not None and highlighter.core() is None:
            highlighter.set_style(barchart_config['bc_red'])
        if titles is not None:
            if not isinstance(titles, Data):
                titles = Data(titles)
            titles.views.append(self)
        self.titles = titles
        if data1 is not None:
            if not isinstance(data1, Data):
                data1 = Data(data1)
            data1.views.append(self)
        self.data1 = data1
        self.show_titles = show_titles
        self.padding = padding
        self.e_idx = e_idx
        # e_idx.views.append(self)

    def generate_vis_data(self):
        r = {}
        data = pd.DataFrame()
        m1, m2 = self.data.value_().max(), self.data.value_().min()
        m3, m4 = self.data1.value_().max(), self.data1.value_().min()
        m1, m2 = max([m1, m2, m3, m4]), min([m1, m2, m3, m4])
        dist = m1 - m2
        m1, m2 = m1+0.05*dist, m2-0.05*dist
        data['height'] = self.size[1] / (m1 - m2) * (self.data.value_() - m2)
        if self.titles is None:
            r['titles'] = list(range(data.shape[0]))
        else:
            r['titles'] = self.titles.value_()
        data['x'] = r['titles']
        data['color'] = '#0095b6'
        if self.e_idx is None or self.e_idx.value_() is None:
            data['idx'] = list(range(data.shape[0]))
        else:
            data['idx'] = self.e_idx.value_()
        r['rect'] = data.to_dict(orient='records')
        # data1
        data = pd.DataFrame()
        data['height'] = self.size[1] / (m1 - m2) * (self.data1.value_() - m2)
        data['x'] = r['titles']
        data['color'] = '#b62100'
        if self.e_idx is None or self.e_idx.value_() is None:
            data['idx'] = list(range(data.shape[0]))
        else:
            data['idx'] = self.e_idx.value_()
        data['color'] = 'red'
        r['path'] = data.to_dict(orient='records')
        r['range'] = [float(m2), float(m1)]
        return json.dumps(r)

    def core(self):
        super(PointChart, self).core()
        View.f.write(f"""
const sx{self.idx} = d3.scaleBand()
    .range([0, rw{self.idx}]).paddingInner(1).paddingOuter(0.2);
const sy{self.idx} = d3.scaleLinear()
    .range([rh{self.idx}, 0]);
// maybe forward to the head() in utils.py
var cross = d3.symbol().type(d3.symbolCross).size(10);
trigger{self.idx}.on('click', () => {{
    d3.json('/trigger/{self.idx}').then(r => {{
        sx{self.idx}.domain(r['titles']);
        sy{self.idx}.domain(r['range']);
        {self.axis()}
        g{self.idx}.selectAll('circle.a').remove();
        g{self.idx}.selectAll('circle.a')
            .data(r['rect']).enter().append('circle')
            .classed('a', true)
            .attr('width', d => sx{self.idx}.bandwidth())
            .attr('height', d => d.height)
            .attr('cx', d => sx{self.idx}(d.x))
            .attr('cy', d => {self.size[1]} - d.height)
            .attr('r', 2)
            .attr('fill', d => d.color)
            .on('click', e => {{
                // all need to record position
                y_click = e.clientY + document.documentElement.scrollTop;
                x_click = e.clientX + document.documentElement.scrollLeft;
                d3.json(`/click/{self.idx}?value=${{{select_this}.datum().idx}}`).then(r => {{
                    for(let i of r[0]) triggers[i].dispatch('click');
                    for(let i of r[1]) highlighters[i].dispatch('click');
                }});
            }});
        g{self.idx}.selectAll('path.a').remove();
        g{self.idx}.selectAll('path.a')
           .data(r['path']).enter().append('path')
           .classed('a', true)
           .attr('fill', d => d.color)
           .attr('transform', d => `translate(${{sx{self.idx}(d.x)}}, ${{{self.size[1]} - d.height}}) rotate(45, 0, 0)`)
           .attr('d', cross);
    }});
}});
trigger{self.idx}.dispatch('click');
highlighter{self.idx}.on('click', e => {{
    d3.json('/highlighter/{self.idx}').then(r => {{
        g{self.idx}.selectAll('circle.a'){self.highlighter.core() if self.highlighter is not None else ';'}
    }});
}});
        """.strip() + "\n")

    def click(self, request_args):
        value = int(request_args.get('value'))
        self.last_value = value
        View.update_list.clear()
        View.highlight_list.clear()
        self.click_(value)
        return json.dumps([View.update_list, View.highlight_list])

    def axis(self):
        return f"""
const ax{self.idx} = d3.axisBottom(sx{self.idx}).tickSizeOuter(0), ay{self.idx} = d3.axisLeft(sy{self.idx}).tickSizeOuter(0).tickSizeInner(4);
g{self.idx}.selectAll('g[class*=axis_]').remove();
g{self.idx}.append('g').call(ax{self.idx}).attr('transform', 'translate(0, {self.size[1]})').attr('fill', 'none')
.classed('axis_x', true)""" + (f".selectAll('.tick').remove()" if self.titles is None else '') + f""";
g{self.idx}.append('g').call(ay{self.idx}).attr('fill', 'none').classed('axis_y', true);
g{self.idx}.selectAll('.domain').attr('stroke', '#d3d3d3');
g{self.idx}.selectAll('.tick line').attr('stroke', '#d3d3d3');
g{self.idx}.selectAll('.tick text').attr('fill', '#708090');
        """.strip() + "\n"


# class LinkMap(View):
#     # 一个特殊的view，需要修正
#     def __init__(self, position, size, data, reg_no=-1, **info):
#         """
#         info: max_height, reg_no
#         """
#         super(LinkMap, self).__init__(position, size, data, reg_no, border=False, info=info)
#         self.elements = Link(self.idx, reg_no=reg_no)
#
#     def core(self, views=None):
#         super(LinkMap, self).core(views)
#         ele_id = self.elements.ele_id
#         # 就差这一个(这是刚加的，所以其实有两个）正向比例尺，后面可能会统一一下
#         View.f.write(f"""
# const sx{ele_id} = d3.scaleLinear()
#     .domain([0, 1000])
#     .range([rx{self.idx}, rx{self.idx} + rw{self.idx}]);
# const sy{ele_id} = d3.scaleLinear()
#     .domain([0, 400])
#     .range([ry{self.idx}, ry{self.idx} + rh{self.idx}]);
#                 """.strip() + "\n")
#
#     def response(self, views=None, back=False):
#         if back:
#             ele_id = self.elements.ele_id
#             return f"""
# if(typeof(selected_back[{self.reg_no}]['view']) != 'undefined'){{
#     const data{self.idx} = selected_back[{self.reg_no}]['view'];
#     g{self.idx}.selectAll('text').remove()
#     g{self.idx}.selectAll('text')
#         .data(data{self.idx})
#         .enter()
#         .append('text')
#         .attr('x', d => sx{ele_id}(d.x))
#         .attr('y', d => sy{ele_id}(d.y))
#         .text(d => d.text)
#         .on('click', e => {{
#             const d = {select_this}.datum();
#             if(d.idx != -1){{
#             const attn = [];
#                 g{self.idx}.selectAll('path')
#                     .attr('stroke', d_ => {{
#                         if(d_.idx>=d.idx*selected_back['il'] && d_.idx<(d.idx+1)*selected_back['il']){{
#                             attn.push(d_.attn);
#                             return 'green';
#                         }}
#                         else return 'grey';
#                     }});
#                 g{self.idx}.selectAll('text.attn').remove();
#                 g{self.idx}.selectAll('text.attn')
#                     .data(attn)
#                     .enter()
#                     .append('text')
#                     .classed('attn', true)
#                     .attr('x', (d, i) => sx{ele_id}(i*200))
#                     .attr('y', (d, i) => sy{ele_id}(380))
#                     .text(d => d.toFixed(2));
#             }}
#         }});
#     {self.elements.response(back)}
# }}
#                 """.strip() + "\n"
#         else:
#             return ''


class Line(View):
    def __init__(self, data, position, size=None, color='black', width=1, opacity=1):
        """
        :param data: matrix type
        if labels, don't use thin size rectangle.
        """
        if not isinstance(data, Data):
            data = Data(data)
        super(Line, self).__init__(data, position, size, border=False)
        self.color = color
        self.width = width
        self.opacity = opacity

    def generate_vis_data(self):
        return json.dumps({'padding':self.data.value_().tolist(), 'color':self.color, 'width':self.width, 'opacity':self.opacity})

    def core(self):
        super(Line, self).core()
        View.f.write(f"""
trigger{self.idx}.on('click', () => {{
    d3.json('/trigger/{self.idx}').then(r => {{
        g{self.idx}.selectAll('path').remove();
        g{self.idx}.append('path')
            .attr('d', d3.line()([[r['padding'][0], 0], [{self.size[0]}-r['padding'][1], 0]]))
            .attr('stroke', r['color'])
            .attr('stroke-width', r['width'])
            .attr('stroke-opacity', r['opacity']);
    }});
}});
trigger{self.idx}.dispatch('click');
        """)


class Title(View):
    def __init__(self, data, position, size=None, color='black', opacity=1, bg_color='gray'):
        """
        :param data: matrix type
        if labels, don't use thin size rectangle.
        """
        if not isinstance(data, Data):
            data = Data(data)
        super(Title, self).__init__(data, position, size, border=False)
        self.color = color
        self.opacity = opacity
        self.bg_color = bg_color

    def generate_vis_data(self):
        return json.dumps({'title':self.data.value_(), 'color':self.color, 'opacity':self.opacity, 'bg_color':self.bg_color})

    def core(self):
        super(Title, self).core()
        View.f.write(f"""
trigger{self.idx}.on('click', () => {{
    d3.json('/trigger/{self.idx}').then(r => {{
        g{self.idx}.selectAll('rect.a').remove();
        g{self.idx}.append('rect')
            .classed('a', 'true')
            .attr('x', 0).attr('y', 0).attr('opacity', r['opacity'])
            .attr('width', {self.size[0]}).attr('height', {self.size[1]})
            .attr('fill', r['bg_color']);
        g{self.idx}.selectAll('text').remove();
        g{self.idx}.append('text')
            .attr('fill', r['color'])
            .attr('x', {self.size[0]} / 2)
            .attr('y', {self.size[1]} / 2)
            .attr('dy', '.35em')
            .attr('text-anchor', 'middle')
            .attr('font-size', 16)
            .text(r['title']);

    }});
}});
trigger{self.idx}.dispatch('click');
        """)


class HeatMap(View):
    def __init__(self, data, position, size=None, title=None, cm=None, cm_highlight=None, cell_size=[20, 25], labels=None,
                 highlighter=None, selector=None, opacity=1):
        """
        :param data: matrix type
        if labels, don't use thin size rectangle.
        """
        super(HeatMap, self).__init__(data, position, size, highlighter=highlighter, title=title, border=False)
        if cm is None:
            self.cm = plt.get_cmap(heat_map_config['cm'])
        elif isinstance(cm, str):
            self.cm = plt.get_cmap(cm)
        else:
            self.cm = cm
        # a trick to simplify code, but easy to improve, and will be improve
        if title is not None and ('part' in title or 'pos' in title):
            self.cm = plt.get_cmap('tab20')
        if cm_highlight is None:
            self.cm_highlight = plt.get_cmap('Reds')
        elif isinstance(cm, str):
            self.cm_highlight = plt.get_cmap(cm_highlight)
        else:
            self.cm_highlight = cm_highlight
        self.cell_size = cell_size
        if highlighter is not None and highlighter.core() is None:
            highlighter.set_style(heat_map_config['style1'])
        self.selector = selector
        self.labels = labels
        self.opacity = opacity
        if labels is not None:
            labels.views.append(self)
        self.shape = None

    def generate_vis_data(self):
        data = pd.DataFrame()
        if self.labels is not None:
            data['label'] = self.labels.value_().reshape(-1)
            if len(self.labels.size()) != 1:
                self.shape = self.labels.size()
            else:
                self.shape = [1, self.labels.size()[0]]
        if self.data is not None:
            value = self.data.value_()
            if isinstance(value, (list, np.ndarray)) and len(value) != 0:
                if np.max(value) != 0:
                    value = value / np.max(value)
                # a trick to simplify code, but easy to improve, and will be improve
                if self.title is not None and ('part' in self.title or 'pos' in self.title):
                    value[value > 19] = 19
                data['color'] = list(
                    map(lambda x: colors.to_hex(self.cm(x), keep_alpha=True), value.reshape(-1).tolist()))
                data['h_color'] = list(
                    map(lambda x: colors.to_hex(self.cm_highlight(x), keep_alpha=True), value.reshape(-1).tolist()))
                if len(self.data.size()) != 1:
                    self.shape = self.data.size()
                else:
                    self.shape = [1, self.data.size()[0]]
        else:
            data['color'] = heat_map_config['color']
        if self.cell_size is None:
            if self.size is not None:
                width = self.size[0] / self.shape[1]
                height = self.size[1] / self.shape[0]
            else:
                width, height = 25, 15
        else:
            width, height = self.cell_size
        data['width'] = width
        data['height'] = height
        data['x'] = [width * j for i in range(self.shape[0]) for j in range(self.shape[1])]
        data['y'] = [height * i for i in range(self.shape[0]) for j in range(self.shape[1])]
        data['opacity'] = self.opacity
        data['idx'] = list(range(data.shape[0]))
        data['idt'] = [[i, j] for i in range(self.shape[0]) for j in range(self.shape[1])]
        return data.to_json(orient='records')

    def core(self):
        super(HeatMap, self).core()
        View.f.write(f"""
trigger{self.idx}.on('click', () => {{
    d3.json('/trigger/{self.idx}').then(r => {{
        g{self.idx}.selectAll('rect.a').remove();
        g{self.idx}.selectAll('rect.a')
            .data(r).enter().append('rect')
            .classed('a', 'true')
            .attr('x', d => d.x).attr('y', d => d.y).attr('opacity', d => d.opacity)
            .attr('width', d => d.width).attr('height', d => d.height)
            .attr('fill', d => {{
                if(typeof(d.color) != 'undefined') return d.color;
                else return 'aqua';
            }})
            .on('click', e => {{
                d3.json(`/click/{self.idx}?value=${{{select_this}.datum().idx}}`).then(r => {{
                    for(let i of r[0]) triggers[i].dispatch('click');
                    for(let i of r[1]) highlighters[i].dispatch('click');
                }});
            }});
        if(r.length != 0 && typeof(r[0].label) != 'undefined'){{
            g{self.idx}.selectAll('text').remove();
            g{self.idx}.selectAll('text')
                .data(r)
                .enter()
                .append('text')
                .attr('fill', 'black')
                .attr('x', d => d.x + d.width / 2)
                .attr('y', d => d.y + d.height / 2)
                .attr('dy', '.35em')
                .attr('text-anchor', 'middle')
                .attr('font-size', d => d.width * 0.6)
                .text(d => d.label)
                .on('click', e => {{
                    d3.json(`/click/{self.idx}?value=${{{select_this}.datum().idx}}`).then(r => {{
                        for(let i of r[0]) triggers[i].dispatch('click');
                        for(let i of r[1]) highlighters[i].dispatch('click');
                    }});
                }});
        }}
    }});
}});
trigger{self.idx}.dispatch('click');
highlighter{self.idx}.on('click', e => {{
    d3.json('/highlighter/{self.idx}').then(r => {{
        g{self.idx}.selectAll('rect.a'){self.highlighter.core() if self.highlighter is not None else ';'}
    }});
}});
        """)

    def click(self, request_args):
        value = int(request_args.get('value'))
        row = value // self.shape[1]
        col = value % self.shape[1]
        View.update_list.clear()
        View.highlight_list.clear()
        if self.click_ is not None:
            if self.selector == 'col':
                self.click_(col)
            elif self.selector == 'row':
                self.click_(row)
            elif self.selector == 'pos':
                self.click_(row, col)
            else:
                self.click_(value)
        return json.dumps([View.update_list, View.highlight_list])


class TextView(View):
    def __init__(self, data, position=[50, 50], size=None, title=None, cell_size=[36, 36], orient='vertical'):
        """
        :param data: Vector type
        """
        super(TextView, self).__init__(data, position, size, title=title, border=False)
        self.cell_size = cell_size

    def text_positions(self, selected=None):
        self.shape = self.data.size()
        if self.cell_size is None:
            if self.size is not None:
                width = self.size[0] / 1
                height = self.size[1] / self.shape[0]
            else:
                width, height = 35, 25
        else:
            width, height = self.cell_size
        if selected is None:
            selected = list(range(self.shape[0]))
        elif not isinstance(selected, list):
            selected = [selected] * self.shape[0]
        else:
            print("Error.")
        return [(i + 0.5) * height + self.position[1] for i in selected]

    def generate_vis_data(self):
        r = {}
        data = pd.DataFrame()
        data['text'] = self.data.value_()
        self.shape = self.data.size()
        if self.cell_size is None:
            if self.size is not None:
                width = self.size[0] / 1
                height = self.size[1] / self.shape[0]
            else:
                width, height = 25, 15
        else:
            width, height = self.cell_size
        # data['font_size'] = [width * 0.3 if len(w) < 6 else 4 / len(w) * width * 0.4 for w in value]
        data['font_size'] = 19
        data['width'] = width
        data['height'] = height
        data['idx'] = list(range(data.shape[0]))
        data['x'] = 0
        data['y'] = [height * i for i in range(self.shape[0])]
        data['color'] = 'black'
        # r['d'] = data.to_dict(orient='records')
        return data.to_json(orient='records')

    def core(self):
        super(TextView, self).core()
        View.f.write(f"""
trigger{self.idx}.on('click', e => {{
    d3.json('/trigger/{self.idx}').then(r => {{
        g{self.idx}.selectAll('text').remove();
        g{self.idx}.selectAll('text')
            .data(r).enter().append('text')
            .attr('fill', 'black')
            .attr('x', d => d.x + d.width / 2)
            .attr('y', d => d.y + d.height / 2)
            .attr('dy', '.35em')
            .attr('text-anchor', 'middle')
            .attr('font-size', d => d.font_size)
            .text(d => d.text)
            .on('click', e => {{
            d3.json(`/click/{self.idx}?value=${{{select_this}.datum().idx}}`).then(r => {{
                    for(let i of r[0]) triggers[i].dispatch('click');
                    for(let i of r[1]) highlighters[i].dispatch('click');
                }});
            }});
    }});
}});
trigger{self.idx}.dispatch('click');
        """)

    def click(self, request_args):
        value = int(request_args.get('value'))
        self.last_value = value
        View.update_list.clear()
        View.highlight_list.clear()
        if self.click_ is not None:
            self.click_(value)
        return json.dumps([View.update_list, View.highlight_list])


class LinkView(View):
    def __init__(self, data=None, position=None, size=None, title=None, orient='horizontal', node_positions=None,
                 labels=None, colors=None, cm=None, width=None):
        """
        :param data: Vector, the width of link, None for 1
        :param labels: Vector
        :param node_positions: Matrix, [start-Vector, end-Vector]
        temp ly designed only for horizontal
        """
        super(LinkView, self).__init__(data, position, size, title=title, border=False)
        # current just for horizontal but easy to add vertical
        self.orient = orient
        self.node_positions = node_positions
        if node_positions is not None:
            node_positions.views.append(self)
        self.labels = labels
        if labels is not None:
            labels.views.append(self)
        self.colors = colors
        self.width = width
        if isinstance(colors, Data):
            colors.views.append(self)
        self.cm = cm
        if cm is None:
            self.cm = link_view_config['cm']
            if isinstance(self.cm, str):
                self.cm = plt.get_cmap(self.cm)

    def generate_vis_data(self):
        data = pd.DataFrame()
        value = self.node_positions.value_() if self.node_positions is not None else self.data.value_()
        ends = []
        for i in range(value.shape[1]):
            ends.append({'source': [0, value[0][i]],
                         'target': [self.size[0], value[1][i]]})
        data['nodes'] = ends
        if self.width is None:
            if self.data is None:
                width = 2
            else:
                width = 5 * self.data.value_()
        elif self.width == 'labels':
            width = np.interp(x=self.labels.value_().__abs__(), xp=[0, 20], fp=[0.5, 10])
        elif isinstance(self.width, Data):
            width = self.width.value_()
        else:
            width = self.width
        data['width'] = width
        if self.labels is not None:
            data['label'] = self.labels.value_()
        if self.colors == 'labels':
            data['color'] = list(map(lambda x: colors.to_hex(self.cm(x), keep_alpha=True),
                                     self.labels.tolist()))
        elif isinstance(self.colors, Data):
            value = self.colors.value_()
            if len(value) != 0:
                if isinstance(value[0], str):
                    data['color'] = value
                else:
                    data['color'] = list(map(lambda x: colors.to_hex(self.cm(x), keep_alpha=True),
                                             value.reshape(-1).tolist()))
        else:
            data['color'] = trans_demo_config['link_color']
        data['idx'] = list(range(data.shape[0]))
        # l = self.data.__value__().shape[0]
        # for i in range(l):
        #     ends.append({'source': [0, 15 + self.size[1] / l * i],
        #                  'target': [self.size[0] - 50, 15 + self.size[1] / l * self.data.info.get('sid', 0)]})
        return data.to_json(orient='records')

    def core(self):
        super(LinkView, self).core()
        View.f.write(f"""
const link{self.idx} = d3.linkHorizontal();
//special position for this
g{self.idx}.attr('transform', `translate(${{rx{self.idx}}}, 0)`);
trigger{self.idx}.on('click', e => {{
    d3.json('/trigger/{self.idx}').then(r => {{
        g{self.idx}.selectAll('path').remove();
        g{self.idx}.selectAll('path')
            .data(r).enter().append('path')
            .attr('d', d => link{self.idx}({{'source':[(d.nodes.source[0]), (d.nodes.source[1])], 
                'target':[(d.nodes.target[0]), (d.nodes.target[1])]}}))
            .attr('stroke', d => d.color)
            .attr('stroke-width', d => {{
                if(d.width > 0.5) return d.width;
                return 0.5;
            }})
            .attr('fill', 'none')
            .on('click', e => {{
                const t = {select_this}, d = t.datum();
                let y_pos = e.clientY + document.documentElement.scrollTop;
                let x_pos = e.clientX + document.documentElement.scrollLeft;
                toolTip2.attr('style', 'left:' + x_pos + 'px' + ';top:' + y_pos + 'px').selectAll('*').remove();
                toolTip2.style('width', '108px').style('height', '88px');
                toolTip2.append('input')
                    .attr('id', 'lr{self.idx}')
                    .attr('type', 'text')
                    .attr('value', d.label)
                    .style('width', '100px')
                    .style('height', '20px');
                toolTip2.append('button')
                    .text('update')
                    .style('top', '44px')
                    .style('left', '6px')
                    .on('click', e => {{
                        const value = document.getElementById('lr{self.idx}').value;
                        d3.json(`/click/{self.idx}?value=${{d.idx}}&w=${{value}}`).then(r => {{
                            for(let i of r[0]) triggers[i].dispatch('click');
                            for(let i of r[1]) highlighters[i].dispatch('click');
                        }});
                        toolTip2.classed('hidden', true);
                    }});
                toolTip2.append('button')
                    .text('close')
                    .style('top', '66px')
                    .style('left', '6px')
                    .on('click', e => {{
                        toolTip2.classed('hidden', true);
                    }});
                toolTip2.classed('hidden', false);
            }});
    }});
}});
trigger{self.idx}.dispatch('click');
        """)

    def click(self, request_args):
        idx = int(request_args.get('value'))
        w = float(request_args.get('w'))
        View.update_list.clear()
        View.highlight_list.clear()
        if self.click_ is not None:
            self.click_(idx, w)

        return json.dumps([View.update_list, View.highlight_list])


# class TransDemo(View):
#     def __init__(self, position, size, data, reg_no=-1, **info):
#         """
#         info: max_height, reg_no
#         """
#         super(TransDemo, self).__init__(position, size, data, reg_no, border=False, info=info)
#         self.words = info['words']
#         self.t_words = info.get('t_words')
#         Container.click_handler['sa'] = self.sa
#         Container.click_handler['h'] = self.h
#         Container.other_handler['select_sen'] = self.select_sen
#         Container.other_handler['select_layer'] = self.select_layer
#         self.selected_word = 0
#         self.selected_layer = 0
#         self.start = -1
#         self.corpus_len = info.get('corpus_len')
#         self.embeddings = info.get('embeddings')
#         self.corpus_words = info.get('corpus_words')
#         self.corpus_refs = info.get('corpus_refs')
#         self.selected_heads = []
#
#     def core(self, views=None):
#         super(TransDemo, self).core(views)
#         # ele_id = self.elements.ele_id
#         # 就差这一个(这是刚加的，所以其实有两个）正向比例尺，后面可能会统一一下
#         View.f.write(f"""
# const g{self.idx}_ = d2.append('div').classed('div3', true);
# g{self.idx}_.style('width', '500px')
#     .style('height', '100px')
#     .style('left', '{self.position[0]}px')
#     .style('top', '{self.position[1] - 125}px');
# const sx{self.idx} = d3.scaleLinear()
#     .domain([0, 800])
#     .range([rx{self.idx}, rx{self.idx} + rw{self.idx}]);
# const sy{self.idx} = d3.scaleLinear()
#     .domain([0, 500])
#     .range([ry{self.idx}, ry{self.idx} + rh{self.idx}]);
# const link{self.idx} = d3.linkHorizontal();
# g{self.idx}_.append('input').attr('type', 'text')
#     .attr('value', '{' '.join(self.words)}')
#     .style('width', '200px')
#     .attr('id', 'input{self.idx}');
# g{self.idx}_.append('button').text('update').classed('update{self.idx}', true)
#     .style('position', 'absolute')
#     .style('top', '2px')
#     .style('left', '250px')
#     .on('click', e => {{
#     const text = document.getElementById('input{self.idx}').value;
#     d3.json(`/select_sen/?sen=${{text}}`).then(r => {{
#         //g{self.idx}_.select('p').remove();
#         //g{self.idx}_.append('p').text(r['t_words']);
#
#         g{self.idx}.selectAll('text').remove();
#         g{self.idx}.selectAll('text.up{self.idx}')
#             .data(r['t_words'])
#             .enter()
#             .append('text')
#             .classed('up{self.idx}', true)
#             .text(d => d.text)
#             .attr('x', d => sx{self.idx}(d.x))
#             .attr('y', d => sy{self.idx}(d.y))
#             .attr('fill', '{trans_demo_config['text_color']}')
#             .on('mouseover', e => {{
#                 const d = {select_this}.datum();
#                 let y_pos = e.clientY + document.documentElement.scrollTop
#                 let x_pos = e.clientX + document.documentElement.scrollLeft
#                 toolTip2.attr('style', 'left:' + x_pos + 'px' + ';top:' + y_pos + 'px').selectAll('*').remove();
#                 toolTip2.append('textarea')
#                     .attr('rows', 16).attr('cols', 20)
#                     .text(d.prob);
#                 toolTip2.classed("hidden", false);
#             }})
#             .on('mouseout', e => {{
#                 toolTip2.classed("hidden", true);
#             }});
#         g{self.idx}.selectAll('text.left{self.idx}')
#             .data(r['support'])
#             .enter()
#             .append('text')
#             .classed('left{self.idx}', true)
#             .attr('x', d => sx{self.idx}(d.x))
#             .attr('y', d => sy{self.idx}(d.y))
#             .attr('fill', '{trans_demo_config['text_color']}')
#             .text(d => d.text)
#             .on('click', e => {{
#                 const d = {select_this}.datum();
#                 if(d.idx != -1){{
#                     d3.json(`/click/sa?idx=${{d.idx}}`).then(r => {{
#                         g{self.idx}.selectAll('rect.sa').remove();
#                         g{self.idx}.selectAll('rect.sa')
#                             .data(r['matrix'])
#                             .enter()
#                             .append('rect')
#                             .classed('sa', true)
#                             .attr('x', d => sx{self.idx}(d.x))
#                             .attr('y', d => sy{self.idx}(d.y))
#                             .attr('width', d => d.width)
#                             .attr('height', d => d.height)
#                             .attr('fill', d => d.color)
#                             .on('click', e => {{
#                                 const idx = {select_this}.datum().idx;
#                                 d3.json(`/click/h?idx=${{idx}}`).then(r => {{
#                                     g{self.idx}.selectAll('path.h').remove();
#                                     g{self.idx}.selectAll('path.h')
#                                         .data(r['matrix'])
#                                         .enter()
#                                         .append('path')
#                                         .classed('h', true)
#                                         .attr('d', d => link{self.idx}({{'source':[sx{self.idx}(d.ends.source[0]), sy{self.idx}(d.ends.source[1])],
#                                             'target':[sx{self.idx}(d.ends.target[0]), sy{self.idx}(d.ends.target[1])]}}))
#                                         .attr('stroke', d => d.color)
#                                         .attr('stroke-width', d => d.width>0.1?`${{d.width}}px`:'0.1px')
#                                         .attr('fill', 'none');
#                                     g{self.idx}_.select('p').remove();
#                                     g{self.idx}_.append('p').text('selected_heads: ' + r['shs']);
#                                 }});
#                             }});
#                         //g{self.idx}.selectAll('rect.sa').filter(d => d.idx == 0).dispatch('click').dispatch('click');
#                         d3.json(`/click/h`).then(r => {{
#                             g{self.idx}.selectAll('path.h').remove();
#                             g{self.idx}.selectAll('path.h')
#                                 .data(r['matrix'])
#                                 .enter()
#                                 .append('path')
#                                 .classed('h', true)
#                                 .attr('d', d => link{self.idx}({{'source':[sx{self.idx}(d.ends.source[0]), sy{self.idx}(d.ends.source[1])],
#                                     'target':[sx{self.idx}(d.ends.target[0]), sy{self.idx}(d.ends.target[1])]}}))
#                                 .attr('stroke', d => d.color)
#                                 .attr('stroke-width', d => d.width>0.1?`${{d.width}}px`:'0.1px')
#                                 .attr('fill', 'none');
#                             g{self.idx}_.select('p').remove();
#                             g{self.idx}_.append('p').text('selected_heads: ' + r['shs']);
#                         }});
#                         g{self.idx}.selectAll('text.right{self.idx}').remove();
#                         g{self.idx}.selectAll('text.right{self.idx}')
#                             .data(r['text'])
#                             .enter()
#                             .append('text')
#                             .classed('right{self.idx}', true)
#                             .text(d => d.text)
#                             .style('background-color', 'blue')
#                             .attr('x', d => `${{d.x}}px`)
#                             .attr('y', d => `${{d.y}}px`)
#                             .attr('fill', d => d.color)
#                             .attr('dx', 0)
#                             .attr('dy', 0)
#                             .on('click', e => {{
#                                 const d = {select_this}.datum();
#
#                             }});
#                     }});
#                 }}
#             }});
#         g{self.idx}.selectAll('text.left{self.idx}').filter(d => d.idx==0).dispatch('click');
#         }});
#     }});
# g{self.idx}_.select('button.update{self.idx}').dispatch('click');
# g{self.idx}_.selectAll('button.layer')
#     .data([0, 1])
#     .enter()
#     .append('button')
#     .classed('layer', true)
#     .style('position', 'absolute')
#     .style('right', '20px')
#     .style('top', d => `${{8+30*d}}px`)
#     .text(d => d)
#     .on('click', e => {{
#         const l = {select_this}.text();
#         d3.json(`/select_layer/?layer=${{l}}`).then(_ => {{
#             g{self.idx}.selectAll('text').filter(d => d.idx == 0).dispatch('click');
#         }});
#     }});
#                 """.strip() + "\n")
#
#     def response(self, views=None, back=False):
#         return ''
#
#     def response_input(self):
#         return f"""
# d3.json(`/select_layer/?layer=${{value}}`).then(_ => {{
#     g{self.idx}.selectAll('text').filter(d => d.idx == 0).dispatch('click');
# }});
#         """
#
#     def select_sen(self, info):
#         sen = info.get('sen').split(' ')
#         print(sen)
#         r = {}
#         self.words, self.t_words, self.data, interval, values, indices = self.info['nvt'].trans_demo(sen)
#         self.words = self.words.split(' ')
#         w_data = pd.DataFrame()
#         w_data['x'] = [0 for j in range(len(self.words))] + [400 for i in range(len(self.words))]
#
#         w_data['y'] = [22 + 30 * j for j in range(len(self.words))] + [22 + 30 * i for i in range(len(self.words))]
#         w_data['text'] = self.words + self.words
#         w_data['idx'] = [-1 for j in range(len(self.words))] + list(range(len(self.words)))
#         self.start = interval[0]
#         r['support'] = w_data.to_dict(orient='records')
#         data = pd.DataFrame()
#         data['x'] = [50 * i for i in range(len(self.t_words))]
#         data['y'] = -12
#         data['text'] = self.t_words
#         data['idx'] = list(range(len(self.t_words)))
#         prob = []
#         for i in range(len(self.t_words)):
#             t = ''
#             for v, i in zip(values[i], indices[i]):
#                 t += '%s: %.3f\n' % (i, v)
#             prob.append(t)
#         data['prob'] = prob
#         r['t_words'] = data.to_dict(orient='records')
#         # data = pd.DataFrame()
#         # temp = torch.zeros((8, len(self.words)))
#         # cm = plt.get_cmap(trans_demo_config['rect_cm'])
#         # for i in range(len(self.words)):
#         #     temp += self.data[self.selected_layer, :, i, :]
#         # data['color'] = list(map(lambda x: colors.to_hex(cm(x), keep_alpha=True), temp.reshape(-1).numpy().tolist()))
#         # data['idx'] = [i for i in range(self.data.shape[1]) for j in range(self.data.shape[2])]
#         # data['x'] = [600 + 20 * i for i in range(self.data.shape[1]) for j in range(self.data.shape[2])]
#         # data['y'] = [150 * j for i in range(self.data.shape[1]) for j in range(self.data.shape[2])]
#         # data['width'] = 20 / 700 * self.size[0]
#         # data['height'] = 30 / 500 * self.size[1]
#         # r['matrix'] = data.to_dict(orient='records')
#         return json.dumps(r)
#
#     def select_layer(self, info):
#         l = int(info.get('layer'))
#         self.selected_layer = l
#         return json.dumps('a')
#
#     def sa(self, info):
#         # select attention(head)
#         idx = int(info.get('idx'))
#         num = int(info.get('num', 6))
#         r = {}
#         self.selected_word = idx
#         data = pd.DataFrame()
#         cm = plt.get_cmap(trans_demo_config['rect_cm'])
#         values = self.data[self.selected_layer, :, idx, :]
#         data['color'] = list(map(lambda x: colors.to_hex(cm(x), keep_alpha=True), values.reshape(-1).numpy().tolist()))
#         data['idx'] = [i for i in range(self.data.shape[1]) for j in range(self.data.shape[2])]
#         data['x'] = [600 + 20 * i for i in range(self.data.shape[1]) for j in range(self.data.shape[2])]
#         data['y'] = [30 * j for i in range(self.data.shape[1]) for j in range(self.data.shape[2])]
#         data['width'] = 20 / 700 * self.size[0]
#         data['height'] = 30 / 500 * self.size[1]
#         r['matrix'] = data.to_dict(orient='records')
#         idx = idx + self.start
#         layer = 'encoder.layers.%d' % self.selected_layer
#         embedding = self.embeddings[layer][0, idx]
#         embeddings = self.embeddings[layer][0, :self.corpus_len]
#         dist = np.linalg.norm(embeddings - embedding, axis=1)
#         ids = dist.argsort()[:num]
#         print(ids)
#         data = pd.DataFrame()
#         x = []
#         y = []
#         t = []
#         color = []
#         for row, i in enumerate(ids):
#             for j in range(len(self.corpus_refs)):
#                 if self.corpus_refs[j] <= i < self.corpus_refs[j + 1]:
#                     b = 800
#                     for k in range(self.corpus_refs[j], self.corpus_refs[j + 1]):
#                         t.append(self.corpus_words[k])
#                         color.append('red' if k == i else 'black')
#                         x.append(b)
#                         b += len(self.corpus_words[k]) * 10 + 10
#                         y.append(row * 30 + 215)
#         data['text'] = t
#         data['x'] = x
#         data['y'] = y
#         data['color'] = color
#         r['text'] = data.to_dict(orient='records')
#         return json.dumps(r)
#
#     def h(self, info):
#         idx = int(info.get('idx', -1))
#         if idx != -1:
#             if idx in self.selected_heads:
#                 self.selected_heads.remove(idx)
#             else:
#                 self.selected_heads.append(idx)
#                 self.selected_heads = sorted(self.selected_heads)
#         data = pd.DataFrame()
#         if len(self.selected_heads) != 0:
#             temp = torch.zeros((7,))
#             for i in self.selected_heads:
#                 temp += self.data[self.selected_layer, i, self.selected_word, :]
#             data['attn'] = temp
#             # data['attn'] = self.data[self.selected_layer, idx, self.selected_word, :]
#             data['idx'] = list(range(data.shape[0]))
#             data['width'] = data['attn'] * 5
#             data['color'] = trans_demo_config['link_color']
#             ends = []
#             for i in range(len(self.words)):
#                 ends.append({'source': [100, 15 + 30 * i], 'target': [390, 15 + 30 * self.selected_word]})
#             data['ends'] = ends
#
#         r = {'matrix': data.to_dict(orient='records'), 'shs': self.selected_heads}
#         return json.dumps(r)


# Particular need to modify
# class CompositeView(View):
#     def __init__(self, position, size, data_s, info_s):
#         super(CompositeView, self).__init__(position, size)
#         self.elements = []
#         self.info_s = info_s
#         self.x_domains = []
#         self.y_domains = []
#         for i, (data, info) in enumerate(zip(data_s, info_s)):
#             self.elements.append(Circle(self.idx, data, ele_id=i))
#             if info['class'] in ['decision_border', 'point']:
#                 self.x_domains.append([data['x'].min(), data['x'].max()])
#                 self.y_domains.append([data['y'].min(), data['y'].max()])
#
#     def core(self, views=None):
#         super(CompositeView, self).core(views)
#         for i in range(len(self.elements)):
#             element = self.elements[i]
#             ele_id = element.ele_id
#             if self.info_s[i]['class'] == 'decision_border':
#                 View.f.write(f"""
# const sx{ele_id} = d3.scaleLinear()
#     .domain({self.x_domains[i]})
#     .range([rx{self.idx}, rx{self.idx} + rw{self.idx}]);
# const sy{ele_id} = d3.scaleLinear()
#     .domain({self.y_domains[i]})
#     .range([ry{self.idx} + rh{self.idx}, ry{self.idx}]);
# {element.core(views)}
#                 """.strip() + "\n")
#             elif self.info_s[i]['class'] == 'point':
#                 View.f.write(f"""
# const sx{ele_id} = d3.scaleLinear()
#     .domain({self.x_domains[i]})
#     .range([rx{self.idx} + 0.025 * rw{self.idx}, rx{self.idx} + 0.975 * rw{self.idx}]);
# const sy{ele_id} = d3.scaleLinear()
#     .domain({self.y_domains[i]})
#     .range([ry{self.idx} + 0.975 * rh{self.idx}, ry{self.idx} + 0.025 * rh{self.idx}]);
# {element.core(views)}
#                 """.strip() + "\n")
#             else:
#                 pass
#
#
# class CompositeView1(View):
#     def __init__(self, position, size, data_s, info_s):
#         super(CompositeView1, self).__init__(position, size)
#         self.data = data_s
#         self.x_domains = [self.data[0]['x'].min(), self.data[0]['x'].max()]
#         self.y_domains = [self.data[0]['y'].min(), self.data[0]['y'].max()]
#
#         self.i = 0
#
#     def core(self, views=None):
#         super(CompositeView1, self).core(views)
#         ele_id = 'e%d_%d' % (self.idx, 0)
#         Container.si[ele_id] = self.select_i
#         View.f.write(f"""
# d4.style('width', '500px')
# .style('height', '100px')
# .style('left', '{self.position[0] + 200}px')
# .style('top', '{self.position[1] - 80}px');
# const sx{ele_id} = d3.scaleLinear()
# .domain({self.x_domains})
# .range([rx{self.idx}, rx{self.idx} + rw{self.idx}]);
# const sy{ele_id} = d3.scaleLinear()
# .domain({self.y_domains})
# .range([ry{self.idx} + rh{self.idx}, ry{self.idx}]);
# d4.append('input').attr('type', 'range')
# .attr('id', 'input{self.idx}')
# .attr('min', '0')
# .attr('max', '{len(self.data) - 1}')
# .style('display', 'none')
# .on('change', () => {{
#     const value = document.getElementById('input{self.idx}').value;
#     d3.json(`/select_i/{ele_id}?i=${{value}}`).then(r => {{
#             g{self.idx}.selectAll('circle.{ele_id}').remove();
#             g{self.idx}.selectAll('circle.{ele_id}')
#                 .data(r)
#                 .enter()
#                 .append('circle')
#                 .classed('{ele_id}', true)
#                 .attr('r', d => d.point_size)
#                 .attr('cx', d => sx{ele_id}(d.x))
#                 .attr('cy', d => sy{ele_id}(d.y))
#                 .attr('fill', d => d.color)
#                 .attr('opacity', d => d.opacity);
#         }});
# }});
#         """.strip() + "\n")
#
#     def select_i(self, info):
#         i = int(info.get('i'))
#         self.i = i
#         return self.data[i][['point_size', 'x', 'y', 'color', 'opacity']].to_json(orient='records')


class SelectorView(View):
    def __init__(self, position, size, data=None, reg_no=-1, **info):
        super(SelectorView, self).__init__(position, size, data, reg_no, border=False, info=info)
        data = pd.DataFrame()
        data['y'] = [50 * i for i in range(len(info['reg_ids']))]
        data['x'] = 30
        data['height'] = 40
        data['width'] = [len(x.ids) / 2 for x in info['reg_ids'].values()]
        data['idx'] = list(range(data.shape[0]))
        self.data = data
        data = pd.DataFrame()
        data['text'] = info['reg_ids'].keys()
        data['x'] = 5
        data['y'] = [50 * i for i in range(len(info['reg_ids']))]
        self.data_ = data

    def core(self, views=None):
        super(SelectorView, self).core(views)
        View.f.write(f"""
g{self.idx}.attr('transform', `translate(${{rx{self.idx}}}, ${{ry{self.idx}}})`);
g{self.idx}.append('rect')
    .attr('x', 0).attr('y', 0)
    .attr('width', rw{self.idx}).attr('height', rh{self.idx})
    .attr("fill", 'white')
    .attr('stroke', '{view_config['border_color']}')
    .on('click', e => {{
        //r = empty_r;
    }});;
const data{self.idx} = {self.data.to_json(orient='records')};
const data_{self.idx} = {self.data_.to_json(orient='records')};
g{self.idx}.selectAll('rect.background{self.idx}')
    .data(data{self.idx})
    .enter()
    .append('rect')
    .classed('background{self.idx}', true)
    .attr('x', d => d.x).attr('y', d => d.y)
    .attr('width', d => d.width).attr('height', d => d.height)
    .attr("fill", 'blue')
    .attr('stroke', 'blue');
g{self.idx}.selectAll('text')
    .data(data_{self.idx})
    .enter()
    .append('text')
    .text(d => d.text)
    .attr('x', d => d.x).attr('y', d => d.y)
    .attr('dx', 0).attr('dy', 25)
    .attr("fill", 'black');
        """)

    def response(self, views=None, back=False):
        return f"""
let data_f{self.idx} = [];
let i_ = 0;
for(let k of {list(View.idm.get_all_reg_no())}){{
    for(let j_ of {'selected_back' if back else 'r'}[k]){{
        data_f{self.idx}.push({{'x':30+j_/2, 'width':1, 'y':50*i_, 'height':40}});
    }}
    i_++;
}}
g{self.idx}.selectAll('rect.front{self.idx}').remove();
g{self.idx}.selectAll('rect.front{self.idx}')
    .data(data_f{self.idx})
    .enter()
    .append('rect')
    .classed('front{self.idx}', true)
    .attr('x', d => d.x).attr('y', d => d.y)
    .attr('width', d => d.width).attr('height', d => d.height)
    .attr("fill", 'red')
    .attr('stroke', 'red');
        """

    def get_selected(self, info):
        pass
