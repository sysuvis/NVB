from NNVisBuilder.Views import View
from NNVisBuilder.backend import Container
from NNVisBuilder.GlobalVariables import *
import json
import re


def is_decimal(s):
    return bool(re.match(r'^-?\d+(\.\d+)?$', s))


class Widget:
    idx = 0

    def __init__(self, position, size=(30, 30), title='', title_vertical=False):
        # **info param can be used.
        if position is not None and len(position) == 4:
            self.position = position[:2]
        else:
            self.position = position
        self.size = size
        self.idx = Widget.idx
        Widget.idx += 1
        self.click_ = lambda value: json.dumps([])
        Container.handler['w%d' % self.idx] = self.click
        View.widgets.append(self)
        self.title = title
        self.title_vertical = title_vertical
        self.value = None

    def core(self):
        View.f.write(f"""
const w{self.idx} = d2.append('div').classed('div4', true).style('position', 'absolute');
w{self.idx}.style('width', '{self.size[0]}px')
    .style('height', '{self.size[1]}px')
    .style('left', '{self.position[0]}px')
    .style('top', '{self.position[1]}px');""" + (f"""
const p{self.idx} = w{self.idx}.append('p')
    .text('{self.title}')
    .style('position', 'absolute')
    .style('height', '20px')
    .style('top', `${{(-35-('{self.title}'.length>10?30:0))}}px`)
    .style('left', '0px'); 
        """ if self.title_vertical else f"""
const p{self.idx} = w{self.idx}.append('p')
    .text('{self.title}')
    .style('position', 'absolute')
    .style('height', '20px')
    .style('top', '-15px');
p{self.idx}.style('left', `-${{p{self.idx}.node().getBoundingClientRect().width+10+('{self.title}'.length>10?24:0) + ('{self.title}'.length>15?42:0)}}px`); 
        """))

    def onclick(self, f):
        self.click_ = f

    def click(self, request_args):
        value = request_args.get('value')
        View.update_list.clear()
        if value.isdigit():
            value = int(value)
        elif is_decimal(value):
            value = float(value)
        self.value = value
        self.click_(value)
        return json.dumps(View.update_list)

    def set_position(self, position):
        if position is not None and len(position) == 4:
            self.position = position[:2]
        else:
            self.position = position

    def align(self, info='', view=None, padding=0):
        position = self.position.copy()
        size = self.size.copy()
        info = info.split('),')
        for s in info:
            a = re.findall('-?\d+\.?\d+', s)
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


class Slider(Widget):
    def __init__(self, position, size=None, range=1, title='', default_value=0):
        super(Slider, self).__init__(position, size, title)
        self.default_value = default_value
        if not isinstance(range, list):
            range = [0, range-1]
        self.range = range
        if self.size is None:
            self.size = [max(self.range[1] * 20 + 20, 100), 20]
        self.value = 0

    def core(self):
        super(Slider, self).core()
        View.f.write(f"""
w{self.idx}.append('input')
    .attr('type', 'range')
    .attr('min', {self.range[0]})
    .attr('max', {self.range[1]})
    .attr('value', 0)
    .classed('slider', true)
    .attr('id', 'w{self.idx}')
    .style('width', '100%')
    .style('height', '100%')
    .on('input', e => {{
        const value = document.getElementById('w{self.idx}').value;
        d3.json(`/widget/{self.idx}?value=${{value}}`).then(r => {{
            for(let i of r) triggers[i].dispatch('click');
        }});
    }});
w{self.idx}.append('button')
    .text('Play')
    .style('position', 'absolute')
    .style('width', '40px')
    .style('height', '21px')
    .style('left', '{self.size[0]+10}px')
    .style('top', '1.5px')
    .on('click', e => {{
        const s = document.getElementById('w{self.idx}');
        let i = {self.range[0]};
        let intervalId = setInterval(() => {{
            s.value = i;
            d3.select(s).dispatch('input');
            i++;
            if(i > {self.range[1]}) clearInterval(intervalId);
        }}, 800);
    }});
document.getElementById('w{self.idx}').value = {self.default_value};
w{self.idx}.select('input').dispatch('input');
        """)


class Input(Widget):
    def __init__(self, position, size=[200, 20], text='', title='', init=True):
        super(Input, self).__init__(position, size, title)
        self.text = text
        self.title = title
        self.init = init

    def core(self):
        super(Input, self).core()
        View.f.write(f"""
w{self.idx}.append('input')
    .attr('type', 'text')
    .attr('value', '')
    .attr('id', 'w{self.idx}')
    .style('width', '100%')
    .style('height', '100%')
    .on('change', e => {{
        const value = document.getElementById('w{self.idx}').value;
        d3.json(`/widget/{self.idx}?value=${{value}}`).then(r => {{
            for(let i of r) triggers[i].dispatch('click');
        }});
    }}); """ + (f"""
d2.selectAll('input#w{self.idx}').attr('value', '{self.text}').dispatch('change');
        """ if self.init else ''))

    def click(self, request_args):
        value = request_args.get('value')
        if is_decimal(value):
            value = float(value)
        self.value = value
        View.update_list.clear()
        self.click_(value)
        return json.dumps(View.update_list)


class Select(Widget):
    def __init__(self, position, size=(30, 30), options=None, title='', default_value=0, title_vertical=False):
        super(Select, self).__init__(position, size, title, title_vertical)
        self.options = list(options)
        self.value = 0
        self.default_value = default_value

    def core(self):
        super(Select, self).core()
        View.f.write(f"""
const select{self.idx} = w{self.idx}.append('select').attr('id', 'w{self.idx}');
const options{self.idx} = {self.options};
select{self.idx}.selectAll('option')
    .data(options{self.idx})
    .enter()
    .append('option')
    .text(d => d);
select{self.idx}
    .on('change', e => {{
        const value = document.getElementById('w{self.idx}').selectedIndex;
        d3.json(`/widget/{self.idx}?value=${{value}}`).then(r => {{
            for(let i of r) triggers[i].dispatch('click');
        }});;
    }});
select{self.idx}.node().selectedIndex = {self.default_value};
select{self.idx}.dispatch('change');
        """)

    def click(self, request_args):
        value = int(request_args.get('value'))
        self.value = value
        View.update_list.clear()
        self.click_(value)
        return json.dumps(View.update_list)


class Button(Widget):
    def __init__(self, position, size=(20, 10), text='', title=''):
        super(Button, self).__init__(position, size, title)
        self.text = text

    def core(self):
        super(Button, self).core()
        View.f.write(f"""
    w{self.idx}.append('button')
        .text('{self.text}')
        .style('width', '100%')
        .style('height', '100%')
        .on('click', e => {{
            d3.json(`/widget/{self.idx}?value={self.text}`).then(r => {{
                for(let i of r) triggers[i].dispatch('click');
            }});
        }});
            """)
