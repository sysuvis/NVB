from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib import colors

# This variable changes by the version of browser
if True:
    select_this = 'd3.select(e.srcElement)'
# select_this = 'd3.select(e.path[0])';
newline = '\n'


def get_colormap2(color_list):
    cmap = LinearSegmentedColormap.from_list(None, color_list, N=256)

    def f(value):
        return colors.to_hex(cmap(float(value)))
    return f


def get_colormap3(color_list):
    cmap1 = LinearSegmentedColormap.from_list(None, color_list[:2], N=256)
    cmap2 = LinearSegmentedColormap.from_list(None, color_list[1:], N=256)

    def f(value):
        if -1 <= value < 0:
            return colors.to_hex(cmap1(float(value)))
        else:
            return colors.to_hex(cmap2(float(value)))
    return f


cmaps = {
    'binary_blue': ListedColormap(['#ADD8E6', '#000080']),
    'd10': ListedColormap(['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a']),
    's2': ListedColormap(['#deebf7', '#3182bd']),
    'q7': ListedColormap(['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02','#a6761d'])
}

styles = {
    'heatmap_cm': f"""
    .attr('fill', d => {{
        if(d.idt[1] == r) return d.h_color;
        return d.color;
    }});
    """,
    'pc': f"""
    .attr('fill', d => {{
        if(r == d.idx) return 'red';
        else return d.color;
    }});;
    ;
    """,
    'sp_m': f"""
    .attr('fill', d => {{
        if(r[3] == d.idx) return 'black';
        if(r[2].length == 0) return d.color;
        if(r[2].indexOf(d.idx) == -1) return 'gray';
        else{{
            if(r[0].indexOf(d.idx) == -1) return 'black';
            return d.color;
        }}
    }})
    .attr('r', d => {{
        if(r[3] == d.idx) return 4.2 * d.r;
        if(r[2].indexOf(d.idx) == -1) return d.r;
        else return 2.5 * d.r;
    }});;
    """,
    'hm_m': f"""
    .attr('fill', d => {{
        if(d.idt[0] == r[0] || d.idt[1] == r[1]) return d.h_color;
        return d.color;
    }});
    """,
    'circle_size': """
    .attr('r', d => {
        if(r.indexOf(d.idx) == -1) return d.r;
        else return 2 * d.r;
    });
    """,
    'circle_size1': """
    .attr('r', d => {
        if(r == d.idx) return 3 * d.r;
        else return d.r;
    });
    """,
    'path_color': """
    .attr('stroke', d => {{
        const pos = r.indexOf(d.idx);
        if(pos != -1){{
            return 'blue';
        }}
        else{{
            return d.color;
        }}
    }});
    """,
    'heat_map_style': """
    .attr('fill', d => {
        if(r.indexOf(d.idt[1]) == -1){
            const rgb = d3.rgb(d.color);  
            const gray = 0.2126 * rgb.r + 0.7152 * rgb.g + 0.0722 * rgb.b;
            return 'rgba(' + gray + ', ' + gray + ', ' + gray + ', 1)'
        }
        else{
            return d.color;
        }
    });
    """,
    'hm_white': """
    .attr('fill', d => {
        if(d.idt[1] == r){
            return 'white';
        }
        else{
            return d.color;
        }
    });
    """,
    'bc_red': """
    .attr('fill', d => {
        if(d.idx == r){
            return 'red';
        }
        else{
            return d.color;
        }
    });
    """
}

circle_config = {
    'r': 3.5,
    'color': '#1f77b4',
    'response_color': 'blue',
    'response_back_color': 'red',
    'opacity': 1,
    'cm': 'Accent',
    'cm1': get_colormap3(['#fc8d59', '#ffffbf', '#91bfdb']),
    'style': """
    .attr('fill', d => {
        if(r.indexOf(d.idx) == -1) return d.color;
        else return 'purple';
    });
    """
}

path_config = {
    'color': 'gray',
    'style': """
    .attr('stroke', d => {
        if(r.indexOf(d.idx) == -1) return d.color;
        else return '#5194AD';
    })
    .attr('stroke-width', d => {
        if(r.indexOf(d.idx) == -1) return 1;
        else return 1.5;
    })
    .attr('opacity', d => {
        if(r.indexOf(d.idx) == -1) return 0.6;
        else return 1.0;
    });
    """,
    'style1': """
    .attr('stroke', d => {
        if(r.indexOf(d.idx) == -1) return d.color;
        else return 'blue';
    });
    """,

}

view_config = {
    'border_color': 'gray',
}

heat_map_config = {
    'color': '#4495D9',
    'cm': 'Blues',
    'style1': """
    .attr('fill', d => {
        if(r.indexOf(d.idx) == -1) return '#EA6039';
        else return d.color;
    });
    """,
    'style2': """
    .attr('fill', d => {
        if(r.indexOf(d.idt[1]) == -1){
            const rgb = d3.rgb(d.color);  
            const gray = 0.2126 * rgb.r + 0.7152 * rgb.g + 0.0722 * rgb.b;
            return gray;
        }
        else{
            return d.color;
        }
    });
    """
}

link_view_config = {
    'cm': circle_config['cm1']
}

barchart_config = {
    'color': 'blue',
    'bc_red': """
    .attr('fill', d => {
        if(d.idx == r){
            return 'red';
        }
        else{
            return d.color;
        }
    });
    """
}

selector_view_config = {

}

trans_demo_config = {
    'text_color': 'black',
    'link_color': '#800080',
    'rect_cm': 'Blues',
}

# explanation for the items
selector_config = {
    'background_color': '#fef9ff',
    'rect_color': '#9F9FED',
    'op_union': 'blue',
    'op_intersect': 'red',
    'op_new': '#9F9FED',
    'global_selector_border': 'gray',
}


def rnn_extract_output(output):
    return output[1]
