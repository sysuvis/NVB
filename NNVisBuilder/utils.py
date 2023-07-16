import json
import torch
import numpy as np
import torch.nn as nn
import pandas as pd
# from NNVisBuilder.Views import View
from NNVisBuilder.backend import Container
from NNVisBuilder.GlobalVariables import *
from sklearn.manifold import TSNE

t_obj = TSNE()


def tsne(value):
    if len(value) < 2:
        return []
    return t_obj.fit_transform(value)
    # # value: tensor or ndarray
    # if len(value.shape) == 2:
    #     return t_obj.fit_transform(value)
    # else:
    #     # don't need to separate
    #     shape = value.shape
    #     r = t_obj.fit_transform(value.reshape((-1, shape[-1])))
    #     return r.reshape((*shape[:-1], r.shape[-1]))


class MSelector:
    def __init__(self, gid, selectors=[], set_modes=[]):
        self.gid = gid
        self.selectors = selectors
        self.named_selectors = None
        self.set_modes = set_modes  # 0: union     1: intersect
        self.i = 0
        self.prev_r = None
        self.prev_r_ = None
        self.flag = False
        Container.other_handler['g%d' % gid] = self.msv
        Container.other_handler['ws%d' % gid] = self.which_selector
        Container.other_handler['ss%d' % gid] = self.switch_selector
        Container.other_handler['sc%d' % gid] = self.set_code
        Container.other_handler['as%d' % gid] = self.add_selector

    def set_selectors(self, named_selectors):
        self.named_selectors = named_selectors

    def msv(self):
        r = {}
        data = pd.DataFrame()
        data['code'] = self.selectors
        data['width'] = 40
        data['height'] = 40
        data['idx'] = list(range(len(self.selectors)))
        data['x'] = [80 * i + 10 for i in range(len(self.selectors))]
        data['y'] = 30
        r['set'] = data.to_dict(orient='records')
        data = pd.DataFrame()
        data['mode'] = self.set_modes
        data['idx'] = list(range(len(self.set_modes)))
        data['cx'] = [80 * i + 70 for i in range(len(self.set_modes))]
        data['cy'] = 80
        data['r'] = 10
        data['color'] = [selector_config['op_intersect'] if x == 1 else selector_config['op_union'] for x in
                         self.set_modes]
        r['op'] = data.to_dict(orient='records')
        data = pd.DataFrame()
        data['mode'] = [View.set_mode]
        data['x'] = 10
        data['y'] = 95
        data['width'] = 80 * len(self.selectors) - 40
        data['height'] = 5
        data['stroke'] = selector_config['global_selector_border']
        data['fill'] = [selector_config['op_new'] if View.set_mode == 0 else
                        selector_config['op_union'] if View.set_mode == 1 else selector_config['op_intersect']]
        r['bar'] = data.to_dict(orient='records')
        return json.dumps(r)

    def which_selector(self):
        code = self.selectors[self.i]
        r = {'i': self.i, 'l': len(self.selectors),
             'mode': self.set_modes[self.i - 1] if len(self.set_modes) > self.i - 1 >= 0 else -1,
             'mode1': View.set_mode}
        if code.strip() in self.named_selectors:
            r['code'] = self.named_selectors[code.strip()]
            r['key'] = code
        else:
            r['code'] = code
        return json.dumps(r)

    def switch_selector(self, info):
        bar = info.get('bar', '0')
        if bar == '0':
            i = int(info.get('i'))
            self.set_modes[i] = 1 - self.set_modes[i]
            return json.dumps(
                selector_config['op_intersect'] if self.set_modes[i] == 1 else selector_config['op_union'])
        else:
            View.set_mode += 1
            if View.set_mode == 3:
                View.set_mode = 0
            return json.dumps(selector_config['op_new'] if View.set_mode == 0 else
                              selector_config['op_union'] if View.set_mode == 1 else selector_config['op_intersect'])

    def set_code(self, info):
        i = int(info.get('i'))
        code = info.get('code')
        self.selectors[i] = code

    def add_selector(self, info):
        if info.get('flag', '0') == '1':
            if len(self.selectors) < 3:
                self.selectors.append('\nbrush_ids = [];')
                self.set_modes.append(0)
        elif info.get('flag', '0') == '0':
            if len(self.selectors) > 0:
                self.selectors.pop()
                self.set_modes.pop()
        self.i = 0

    def merge_r(self, r_):
        if self.i == 0:
            r = r_
            self.prev_r_ = r.copy()
            self.flag = False
        elif self.set_modes[self.i - 1] == 0:
            r = View.idm.empty_r()
            for k in r_:
                if isinstance(r_[k], list):
                    r[k] = list(set(r_[k]).union(set(self.prev_r.get(k, []))))
                else:
                    r[k] = r_[k]
        elif self.set_modes[self.i - 1] == 1:
            r = View.idm.empty_r()
            for k in r_:
                if isinstance(r_[k], list):
                    r[k] = list(set(r_[k]).intersection(set(self.prev_r.get(k, []))))
                else:
                    r[k] = r_[k]
            self.flag = True
        if self.i != 0:
            r1 = View.idm.empty_r()
            for k in r_:
                if isinstance(r_[k], list):
                    r1[k] = list(set(r_[k]).union(set(self.prev_r_.get(k, []))))
                else:
                    r1[k] = r_[k]
            self.prev_r_ = r1.copy()
            if self.flag:
                r[2000] = self.prev_r_
        self.prev_r = r.copy()
        self.i += 1
        if self.i == len(self.selectors):
            self.i = 0
            return View.merge_r(r)
        return r

    def core(self):
        return f"""
let svf{self.gid} = false, msi{self.gid} = 0;
const dv{self.gid} = d2.append('div').classed('div4', true);
dv{self.gid}.style('width', '250px')
    .style('height', '150px')
    .style('left', `${{rx{self.gid}+rw{self.gid}-250}}px`)
    .style('top', `${{ry{self.gid}-200}}px`)
    .classed('hidden', true);
let gv{self.gid}, last_r1{self.gid}, reset{self.gid} = true;
g{self.gid}.append('circle')
    .attr('id', 'csv{self.gid}')
    .attr('cx', rx{self.gid}+rw{self.gid}).attr('cy', ry{self.gid}-10)
    .attr('r', 5)
    .attr("fill", 'blue')
    .on('click', e => {{
        if(svf{self.gid}){{
            svf{self.gid} = false;
            dv{self.gid}.classed('hidden', true);
        }}
        else{{
            svf{self.gid} = true;
            dv{self.gid}.classed('hidden', false);
            if(reset{self.gid}){{
                reset{self.gid} = false;
                dv{self.gid}.select('svg').remove();
                gv{self.gid} = dv{self.gid}.append('svg')
                gv{self.gid}.append('rect')
                    .classed('sv{self.gid}', true)
                    .attr('x', 0).attr('y', 0)
                    .attr('width', 250).attr('height', 100)
                    .attr("fill", '{selector_config['background_color']}')
                    .attr('stroke', 'black')
                    .attr('opacity', 0.5);
                d3.json('/msv/{self.gid}').then(r1 => {{
                    gv{self.gid}.selectAll('rect.set{self.gid}')
                        .data(r1['set']).enter().append('rect')
                        .classed('set{self.gid}', true)
                        .attr('x', d => d.x).attr('y', d => d.y)
                        .attr('width', d => d.width).attr('height', d => d.height)
                        .attr('fill', '{selector_config['background_color']}').attr('stroke', 'black')
                        .style('stroke-dasharray', (d, i) => i == 0? ('3, 3'): 'none')
                        .attr('opacity', 0.5)
                        .on('click', e => {{
                            const t = {select_this}, d = t.datum();
                            let y_pos = e.clientY + document.documentElement.scrollTop;
                            let x_pos = e.clientX + document.documentElement.scrollLeft;
                            toolTip2.attr('style', 'left:' + x_pos + 'px' + ';top:' + y_pos + 'px').selectAll('*').remove();
                            toolTip2.append('textarea').attr('id', 'ta{self.gid}')
                                .attr('rows', 30).attr('cols', 60)
                                .text('\\n\\n\\n' + (typeof(d.key) == 'undefined'? d.code: d.key));
                            toolTip2.append('button')
                                .text('update')
                                .on('click', e => {{
                                    const code = document.getElementById('ta{self.gid}').value;
                                    d3.json(`/set_code/{self.gid}?i=${{d.idx}}&code=${{code}}`);
                                    d.code = code;
                                    t.datum(d);
                                    toolTip2.classed('hidden', true);
                                }});
                            toolTip2.append('button')
                                .text('close')
                                .style('left', '100px')
                                .on('click', e => {{
                                    toolTip2.classed('hidden', true);
                                }});
                            toolTip2.classed('hidden', false);
                        }});
                    gv{self.gid}.selectAll('circle.set{self.gid}')
                        .data(r1['op']).enter().append('circle')
                        .classed('set{self.gid}', true)
                        .attr('cx', d => d.cx).attr('cy', d => d.cy)
                        .attr('r', d => d.r)
                        .attr('fill', d => d.color)
                        .on('click', e => {{
                            const t = {select_this};
                            d3.json(`/switch_selector/{self.gid}?i=${{t.datum().idx}}`).then(r_ => {{;
                                t.attr('fill', r_);
                            }});
                        }});
                    gv{self.gid}.selectAll('rect.bar')
                        .data(r1['bar']).enter().append('rect')
                        .classed('bar', true)
                        .attr('x', d => d.x).attr('y', d => d.y)
                        .attr('width', d => d.width).attr('height', d => d.height)
                        .attr('stroke', d => d.stroke).attr('fill', d => d.fill)
                        .on('click', e => {{
                            d3.json(`/switch_selector/{self.gid}?bar=1`).then(r_ => {{;
                                d3.selectAll('rect.bar').attr('fill', r_);
                            }});
                        }});
                }});
            }}
        }}
    }});
dv{self.gid}.selectAll('img.mod{self.gid}')
                .data([{{'idx':0, 'icon':'minus.png'}}, {{'idx':1, 'icon':'plus.png'}}]).enter().append('img')
                .classed('mod{self.gid}', true)
                .attr('src', d => `/static/icon/${{d.icon}}`)
                .attr('width', '10%')
                .on('click', e => {{
                    const flag = {select_this}.datum().idx;
                    d3.json(`/add_selector/{self.gid}?flag=${{flag}}`).then(_ => {{
                        reset{self.gid} = true;
                        const c = g{self.gid}.select('#csv{self.gid}');
                        c.dispatch('click');
                        c.dispatch('click');
                    }});
                }});
g{self.gid}.select('#csv{self.gid}').dispatch('click').dispatch('click');
        """

    def select_core(self):
        # 后面缺的大括号在view中补上，这需要在后面修改成更好的模式
        return f"""
d3.json('/which_selector/{self.gid}').then(r1 => {{
    last_r1{self.gid} = r1;
    eval(r1['code']);
    if(r1['mode'] == -1){{
        gv{self.gid}.selectAll('rect.set{self.gid}')
            .attr('fill', (d, i) => {{
                if(i==0) return '{selector_config['rect_color']}';
                else return '{selector_config['background_color']}';
            }})
            .attr('x', d => d.x);
    }}
    else{{
        gv{self.gid}.selectAll('rect.set{self.gid}')
            .attr('fill', (d, i) => {{
                if(i <= r1['i']) return '{selector_config['rect_color']}';
                else return '{selector_config['background_color']}';
            }});
        if(r1['mode'] == 0){{
            if(r1['l'] < 3){{
                gv{self.gid}.selectAll('rect.set{self.gid}')
                    .transition()
                    .attr('x', (d, i) => {{
                        if(i <= r1['i']){{
                            return 10 + 40 * i;
                        }}
                        else{{
                            return d.x;
                        }}
                    }});
            }}
            else{{
                if(r1['i'] == 1){{
                    gv{self.gid}.selectAll('rect.set{self.gid}')
                        .filter((d, i) => i == 0)
                        .transition()
                        .attr('x', 50);
                }}
                else{{
                    gv{self.gid}.selectAll('rect.set{self.gid}')
                        .filter((d, i) => i == 2)
                        .transition()
                        .attr('x', 130);
                }}
            }}
        }}
        else if(r1['mode'] == 1){{
            if(r1['l'] < 3){{
                gv{self.gid}.selectAll('rect.set{self.gid}')
                    .transition()
                    .attr('x', 45);
            }}
            else{{
                if(r1['i'] == 1){{
                    gv{self.gid}.selectAll('rect.set{self.gid}')
                        .filter((d, i) => i == 0)
                        .transition()
                        .attr('x', 90);
                }}
                else{{
                    gv{self.gid}.selectAll('rect.set{self.gid}')
                        .filter((d, i) => i == 2)
                        .transition()
                        .attr('x', 90);
                }}
            }}
        }}
    }}
    if(r1['i'] < r1['l'] - 1){{
        gv{self.gid}.selectAll('rect.set{self.gid}')
            .attr('stroke', (d, i) => i <= r1['i']? 'none': 'black')
            .style('stroke-dasharray', (d, i) => i == r1['i']+1? ('3, 3'): 'none');
    }}
    else{{
        setTimeout(() => {{
            gv{self.gid}.selectAll('rect.set{self.gid}')
                .attr('fill', '{selector_config['background_color']}').attr('stroke', 'black')
                .attr('x', d => d.x)
                .style('stroke-dasharray', (d, i) => i == 0? ('3, 3'): 'none');
        }}, 500);
        gva.select('rect:last-child')
            .attr('stroke', 'none')
            .attr('fill', '{selector_config['rect_color']}')
            .transition()
            .attr('x', () => {{
                if(union_pos == 10){{
                    union_pos = 50;
                    return 10;
                }}
                else if(r1['mode1'] == 0){{
                    union_pos = 10;
                    gva.selectAll('rect').filter((d, i) => i > 0).remove();
                    gva.append('rect')
                        .attr('x', 10).attr('y', 10)
                        .attr('width', 40).attr('height', 40)
                        .attr('fill', '{selector_config['rect_color']}')
                        .attr('opacity', 0.5);
                }}
                else if(r1['mode1'] == 1){{
                    union_pos += 40;
                    gva.append('rect')
                        .attr('x', union_pos+20).attr('y', 10)
                        .attr('width', 40).attr('height', 40)
                        .attr('fill', '{selector_config['background_color']}').attr('stroke', 'black')
                        .style('stroke-dasharray', ('3, 3'))
                        .attr('opacity', 0.5);
                    return union_pos - 40;
                }}
                else if(r1['mode1'] == 2){{
                    gva.append('rect')
                        .attr('x', union_pos+20).attr('y', 10)
                        .attr('width', 40).attr('height', 40)
                        .attr('fill', '{selector_config['background_color']}').attr('stroke', 'black')
                        .style('stroke-dasharray', ('3, 3'))
                        .attr('opacity', 0.5);
                    return 10;
                }}
            }})
    }}
        """


# class Align:
#     def __init__(self, orient='vertical'):
#         self.orient = orient
#         self.center = None
#         self.reference = None
#
#
# class AlignTo:
#     def __init__(self):
#         self.reference = []
#         self.f = None


class Wrapper(nn.Module):
    def __init__(self):
        super(Wrapper, self).__init__()
        pass

    def forward(self, x):
        return x


class ViewComponent:
    def __init__(self, layer, view_class, transform, position, size, name, view_info):
        self.layer = layer
        self.view_class = view_class
        self.transform = transform
        self.position = position
        self.size = size
        self.name = name
        self.view_info = view_info


def df_tolist(v):
    if isinstance(v, torch.Tensor):
        return v.numpy().tolist()
    elif isinstance(v, np.ndarray):
        return v.tolist()
    else:
        return v


def head(f, views=None, size=[2000, 1200]):
    f.write("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Title</title>
</head>
<style>
    /*name abc... is preserved.*/
    
    #div1 {
        width: 200px;
        position: absolute;
        left: 0px;
        background-color: gainsboro;
    }

    button.trigger{
        display: none;
    }

    #button1 {
        position: absolute;
        top: 5px;
        left: 10px;
        width: 80px;
    }

    #button2 {
        position: absolute;
        top: 55px;
        left: 10px;
        width: 80px;
    }

    #button3 {
        position: absolute;
        top: 105px;
        left: 10px;
        width: 100px;
    }

    #button_e {
        position: absolute;
        top: 2px;
        right: 2px;
        width: 25px;
        height: 20px;
        padding: 1px;
    }

    #select1{
        position: absolute;
        top: 56px;
        left: 130px;
    }

    #select2{
        position: absolute;
        top: 106px;
        left: 120px;
    }

    #div2 {
        position: absolute;
        left: 200px;
        /*background-color: blue;*/
    }

    .div3 {
        /*display: flex;*/
        flex-wrap: wrap;
        /*padding: 0 4px;*/
        width: 0px;
        height: 600px;
        position: absolute;
        left: 600px;
        overflow: auto;
    }
    .div3:after {
        /*content: "";*/
        display: table;
        clear: both;
    }

    .div4 {
        position: absolute;
        top: 0px;
        left: 0px;
    }
    .div4.hidden {
        display: none;
    }

    .toolTip {
        position: absolute;
        width: auto;
        height: auto;
        padding: 5px;
        background-color: rgba(240, 248, 255, 0.8);
        border: 1px solid #ccc;
        -webkit-border-radius: 10px;
        -moz-border-radius: 10px;
        border-radius: 10px;
        font-style: 12px;
        -webkit-box-shadow: 4px 4px 10px rbga(0, 0, 0, 0.4);
        -moz-box-shadow: 4px 4px 10px rbga(0, 0, 0, 0.4);
        box-shadow: 4px 4px 10px rbga(0, 0, 0, 0.4);
        /*pointer-events: none;*/
    }
    .toolTip.hidden {
        display: none;
    }
    .toolTip p {
        margin: 0;
        font-family: sans-serif;
        font-size: 16px;
        line-height: 20px;
    }
    .toolTip button{
        position: absolute;
        left: 2px;
        width: 100px;
    }

    .slider {
        -webkit-appearance: none;
        width: 100%;
        height: 10px;
        background: #ddd;
        outline: none;
        opacity: 0.7;
        -webkit-transition: .2s;
        transition: opacity .2s;
    }

    .slider::-webkit-slider-thumb {
        -webkit-appearance: none;
        appearance: none;
        width: 20px;
        height: 20px;
        background: #4CAF50;
        cursor: pointer;
    }

    .slider::-moz-range-thumb {
        width: 20px;
        height: 20px;
        background: #4CAF50;
        cursor: pointer;
    }

</style>
<body>
""" + f"""
<div id="div1" style="height:{size[1]}px">
    <button id="button1">Tooltip</button>
    <button id="button2">Click</button>
    <button id="button3">SelectorView</button>
    <button id="button_e"><|></button>
    <select id="select1">
        <option>1</option>
        <option>2</option>
    </select>
    <select id="select2">
        <option>New</option>
        <option>Union</option>
        <option>Intersect</option>
    </select>
</div>
<div id="div2">
    <svg width="{size[0]}" , height="{size[1]}"></svg>
    <div class="toolTip" id="toolTip1"></div>
</div>
""" + """
<div class="div3" id="div3"></div>
<div class="div3" id="div4"></div>
{#<iframe src="" width="0" height="0" frameborder="0" name="frame" style="display:none" ></iframe>#}
<div class="toolTip" id="toolTip"></div>
<div class="toolTip" id="toolTip2"></div>
</body>
<script src="/static/d3.js" charset="utf-8"></script>
<script>
    const d1 = d3.select('#div1'), d2 = d3.select('#div2'), svg = d2.select('svg');
    const toolTip = d3.select('#toolTip'), toolTip1 = d3.select('#toolTip1'), toolTip2 = d3.select('#toolTip2');
    const be = document.getElementById('button_e');
    let x_click, y_click;
    be.addEventListener('click', () => {
        if (d1.style('left') == '0px') {
            d1.transition().style('left', '-200px');
            d2.transition().style('left', '0px');
        } else {
            d1.transition().style('left', '0px');
            d2.transition().style('left', '200px');
        }
    });

    const ifTip = document.getElementById('button1');
    const mode = document.getElementById('button2');
    const filter = document.getElementById('select1');
    const filter2 = document.getElementById('select2'), filter2_ = d3.select(filter2);
    const ifSV = document.getElementById('button3');
    /*for (let bu of [be, ifTip, mode, filter, filter2, ifSV]){
        bu.style.display = 'none';
    }*/

    let comp_t = false;
    ifSV.addEventListener('click', () => {
        if (ifSV.style.backgroundColor == 'aqua') {
            ifSV.style.backgroundColor = 'white';
            comp_t = false;
        } else {
            ifSV.style.backgroundColor = 'aqua';
            comp_t = true;
        }
    });

    let tipSwitch = [e => {
        const idx = """ + select_this + """.datum().idx;
        let y_pos = e.clientY + document.documentElement.scrollTop;
        let x_pos = e.clientX + document.documentElement.scrollLeft;
        toolTip.attr('style', 'left:' + x_pos + 'px' + ';top:' + y_pos + 'px').selectAll('*').remove();
        toolTip.append('img')
            .attr('src', `static/img/${idx}.png`)
            .attr('width', 100).attr('height', 100);
        toolTip.classed('hidden', false);
    }, () => {
        toolTip.classed('hidden', true);
    }];
    ifTip.addEventListener('click', () => {
        if(mode.textContent == 'Brush') return;
        if (ifTip.style.backgroundColor == 'aqua') {
            ifTip.style.backgroundColor = 'white'
            d3.selectAll('circle')
                .on('mouseover', null)
                .on('mouseout', null);
        } else {
            ifTip.style.backgroundColor = 'aqua'
            d3.selectAll('circle')
                .on('mouseover', tipSwitch[0])
                .on('mouseout', tipSwitch[1]);
        }
    });""" + f"""
    mode.addEventListener("click", () => {{
        if(mode.textContent == 'Brush'){{
            mode.textContent = 'Click';
            if(typeof(brushG) != 'undefined') d3.brush().clear(brushG);
            {newline.join([v.unbind_brush() for v in views])}
        }}
        else{{
            if(ifTip.style.backgroundColor == 'aqua') ifTip.click();
            mode.textContent = 'Brush';
            {newline.join([v.bind_brush() for v in views])}
        }}
    }});

    filter.addEventListener("change", () => {{
        const idx = filter.selectedIndex;
        d3.json(`/set_click_mode/${{idx}}`);
    }});
    filter2.addEventListener("change", () => {{
        const idx = filter2.selectedIndex;
        d3.json(`/set_set_mode/${{idx}}`);
    }});

    let det; //debug temp variable
    let triggers = new Array();
    let highlighters = new Array();
    let brushG;
    let brush_ids = new Array();
    let empty_r;
    let prev_r_;

    let gva = svg.append('g').attr('transform', 'translate(200, 0)'), union_pos = 10;
    gva.style('display', 'none');
    gva.append('rect')
        .attr('x', 0).attr('y', 0)
        .attr('width', 1000).attr('height', 60)
        .attr("fill", '{selector_config['background_color']}')
        .attr('stroke', 'black')
        .attr('opacity', 0.5);
    gva.append('rect')
        .attr('x', 10).attr('y', 10)
        .attr('width', 40).attr('height', 40)
        .attr('fill', '{selector_config['background_color']}').attr('stroke', 'black')
        .style('stroke-dasharray', ('3, 3'))
        .attr('opacity', 0.5);
    """.strip() + "\n")
