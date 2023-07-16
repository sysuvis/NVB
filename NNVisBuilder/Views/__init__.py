import pandas as pd
from NNVisBuilder.IDManager import IDManager
from NNVisBuilder.GlobalVariables import *
from NNVisBuilder.backend import Container
import re
import json


class View:
    f = None
    views = []
    widgets = []
    update_list = []
    highlight_list = []
    idx = 0
    click_mode = 1
    set_mode = 0  # 0 or others: new set, 1: union, 2: intersect
    idm = IDManager()
    prev_r = {}
    info = {}

    @classmethod
    def set_file(cls, file):
        View.f = file

    @classmethod
    def set_click_mode(cls, mode):
        View.click_mode = mode

    @classmethod
    def set_set_mode(cls, mode):
        View.set_mode = mode

    @classmethod
    def init_prev_r(cls):
        View.prev_r = View.idm.empty_r()

    @classmethod
    def merge_r(cls, r_):
        if View.set_mode == 1:
            r = View.idm.empty_r()
            for k in r_:
                if isinstance(r_[k], list):
                    r[k] = list(set(r_[k]).union(set(View.prev_r.get(k, []))))
                else:
                    r[k] = r_[k]
        elif View.set_mode == 2:
            if View.idm.check_empty_r(View.prev_r):
                r = r_
            else:
                r = View.idm.empty_r()
                for k in r_:
                    if isinstance(r_[k], list):
                        r[k] = list(set(r_[k]).intersection(set(View.prev_r.get(k, []))))
                    else:
                        r[k] = r_[k]
        else:
            r = r_
        View.prev_r = r.copy()
        return r

    @classmethod
    def brush_end_tooltip(cls, views=None):
        return f"""
        const y_pos = y0, x_pos = x1;
        toolTip1.attr('style', 'left:' + x_pos + 'px' + ';top:' + y_pos + 'px').selectAll('*').remove();
        if(comp_t){{
            toolTip1.append('button')
                .text('New')
                .style('top', '25px')
                .on('click', e => {{
                    filter2.selectedIndex = 0;
                    filter2_.dispatch('change');
                    toolTip1.classed('hidden', true);
                    d3.brush().clear(brushG);
                }});
            toolTip1.append('button')
                .text('Union')
                .style('top', '45px')
                .on('click', e => {{
                    filter2.selectedIndex = 1;
                    filter2_.dispatch('change');
                    toolTip1.classed('hidden', true);
                    d3.brush().clear(brushG);
                }});
            toolTip1.append('button')
                .text('Intersect')
                .style('top', '65px')
                .on('click', e => {{
                    filter2.selectedIndex = 2;
                    filter2_.dispatch('change');
                    toolTip1.classed('hidden', true);
                    d3.brush().clear(brushG);
                }});
        }}
        toolTip1.append('button')
            .text('Clear')
            .style('top', '5px')
            .on('click', e => {{
                const temp = filter2.selectedIndex;
                filter2.selectedIndex = 0;
                filter2_.dispatch('change');
                r = empty_r;
                {newline.join([v.response(views) for v in views])}
                d3.json(`/reset_prev_r/`);
                filter2.selectedIndex = temp;
                filter2_.dispatch('change');
                toolTip1.classed('hidden', true);
                d3.brush().clear(brushG);
            }});
        toolTip1.classed('hidden', false);
        """

    def __init__(self, data, position, size=None, highlighter=None, title=None, reg_no=-1, stroke_width=1,
                 stroke_color=view_config['border_color'], border=True, info={}, opacity=1, color='none', stroke_opacity=0.1):
        self.idx = View.idx
        View.idx += 1
        if position is not None and len(position) == 4:
            self.position = position[:2]
            self.size = position[2:]
        else:
            self.position = position
            self.size = None
        if size is None:
            if self.size is None:
                self.size = [100, 100]
        else:
            self.size = size
        self.data = data
        self.highlighter = highlighter
        if self.highlighter is not None:
            self.highlighter.views.append(self)
        self.reg_no = reg_no
        self.title = title
        self.stroke_color = stroke_color
        self.wrap = border
        self.ms = None
        self.info = info
        self.opacity = opacity
        self.color = color
        self.stroke_opacity = stroke_opacity
        self.stroke_width = stroke_width
        self.last_value = 0
        Container.handler['trigger%d' % self.idx] = self.generate_vis_data
        Container.handler['highlight%d' % self.idx] = self.highlight
        Container.handler['c%d' % self.idx] = self.click
        Container.handler['b%d' % self.idx] = self.brush
        if data is not None:
            # if isinstance(data, Data) will be better, but this helps discover errors.
            data.views.append(self)
        View.views.append(self)
        self.click_ = (lambda value: self.highlighter.update(value)) if self.highlighter is not None else None
        self.brush_ = (lambda value: self.highlighter.update(value)) if self.highlighter is not None else None

    def set_position(self, position):
        if len(position) == 2:
            self.position = position.copy()
        elif len(position) == 4:
            self.position = position[:2]
            self.size = position[2:]

    def core(self):
        View.f.write(f"""
const rx{self.idx} = {self.position[0]}, ry{self.idx} = {self.position[1]}, rw{self.idx} = {self.size[0]}, rh{self.idx} = {self.size[1]};
const g{self.idx} = svg.append('g').attr('transform', `translate(${{rx{self.idx}}}, ${{ry{self.idx}}})`).attr('idx', {self.idx});
const trigger{self.idx} = d1.append('button').classed('trigger', true);
const highlighter{self.idx} = d1.append('button').classed('trigger', true);
triggers.push(trigger{self.idx});
highlighters.push(highlighter{self.idx});
let extent{self.idx} = [[0, 0], [rw{self.idx}, rh{self.idx}]];
const brush{self.idx} = d3.brush().on("start", () => {{
        if(brushG != g{self.idx}){{
            if(typeof(brushG) != "undefined") d3.brush().clear(brushG);
            brushG = g{self.idx};
        }}
    }})
    .on("end", e => {{
        if(e.selection){{
            const [[x0, y0], [x1, y1]] = e.selection;
            //...
        }}
        else{{
            r = empty_r;
        }}
    }});
    """ + (f"""
g{self.idx}.append('rect')
    .classed('border_', true)
    .attr('x', 0).attr('y', 0)
    .attr('opacity', {self.opacity})
    .attr('width', rw{self.idx}).attr('height', rh{self.idx})
    .attr("fill", '{self.color}')
    .attr('stroke', '{self.stroke_color}')
    .attr('stroke-width', {self.stroke_width})
    .attr('stroke-opacity', {self.stroke_opacity})
    .on('click', e => {{
        d3.json(`/click/{self.idx}?value=-1`).then(r => {{
            for(let i of r[0]) triggers[i].dispatch('click');
            for(let i of r[1]) highlighters[i].dispatch('click');
        }});
    }});
        """ if self.wrap else '') + (self.ms.core() if self.ms is not None else '') +
                     (f"""
g{self.idx}.append('text')
    .classed('title_', true)
    .attr('fill', 'black')
    .attr('x', -18)
    .attr('y', -10)
    //.attr('text-anchor', 'middle')
    .attr('font-size', '12')
    .text('{self.title}');
                     """ if self.title is not None else '')
                     )

    def axis(self):
        return f"""
const ax{self.idx} = d3.axisBottom(sx{self.idx}), ay{self.idx} = d3.axisLeft(sy{self.idx});
g{self.idx}.selectAll('g[class*=axis_]').remove();
g{self.idx}.append('g').call(ax{self.idx}).attr('transform', 'translate(0, {self.size[1]})').attr('fill', 'none').classed('axis_x', true);
g{self.idx}.append('g').call(ay{self.idx}).attr('fill', 'none').classed('axis_y', true);
        """.strip() + "\n"

    def generate_vis_data(self):
        # two type of return: json of Dataframe or json of dict 'r' including Dataframe and others
        pass

    def click(self, request_args):
        pass

    def onclick(self, f):
        self.click_ = f

    def trigger_click(self):
        self.click_(self.last_value)

    def brush(self, request_args):
        pass

    def on_brush(self, f):
        self.brush_ = f

    def highlight(self):
        return json.dumps(self.highlighter.value) if self.highlighter is not None else json.dumps([])

    def align(self, info='', view=None, padding=0):
        position = self.position.copy()
        size = self.size.copy()
        info = info.split('),')
        for s in info:
            a = re.findall('-?\d+', s)
            if 'right' in s:
                if len(a) != 0:
                    position[0] += int(float(a[0]))
                if 'next' in s:
                    position[0] += self.size[0]
                if view is not None:
                    size[0] = max(view.position[0] - padding - position[0], 0)
            elif 'under' in s:
                if len(a) != 0:
                    position[1] += int(float(a[0]))
                if 'next' in s:
                    position[1] += self.size[1]
                if view is not None:
                    size[1] = max(view.position[1] - padding - position[1], 0)
        return [position[0], position[1], size[0], size[1]]

    def response(self, views=None, back=False):
        return ''

    def response_input(self):
        return ''

    #     def brush(self, views=None):
    #         return ''
    #
    def bind_brush(self):
        return f"""
g{self.idx}.selectAll('rect[class*=overlay]').remove();
g{self.idx}.selectAll('rect[class*=handle]').remove();
g{self.idx}.call(brush{self.idx}.extent(extent{self.idx}));
        """.strip()

    def unbind_brush(self):
        return f"""
g{self.idx}.call(brush{self.idx}.extent([[0, 0], [0, 0]]));
        """.strip()


def set_set_mode(mode):
    View.set_set_mode(mode)


def init_prev_r():
    View.init_prev_r()


Container.other_handler['set_set_mode'] = set_set_mode
Container.other_handler['init_prev_r'] = init_prev_r
