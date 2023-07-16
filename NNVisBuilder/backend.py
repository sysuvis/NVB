from flask import Flask, render_template, request, jsonify
import webbrowser
import threading

app = Flask(__name__)


class Container:
    # One handler is ok.
    handler = {}
    # no need, just history
    click_handler = {}
    brush_handler = {}
    trigger_handler = {}
    other_handler = {}
    si = {}


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/trigger/<int:idx>')
def trigger(idx):
    r = Container.handler['trigger%d' % idx]()
    return r


@app.route('/highlighter/<int:idx>')
def highlighter(idx):
    r = Container.handler['highlight%d' % idx]()
    return r


# click event for widgets
@app.route('/widget/<int:wid>')
def widget_click(wid):
    r = Container.handler['w%d' % wid](request.args)
    return r


# click event for views
@app.route('/click/<int:idx>')
def click(idx):
    r = Container.handler['c%d' % idx](request.args)
    return r


@app.route('/brush/<int:idx>')
def brush(idx):
    r = Container.handler['b%d' % idx](request.args)
    return r


@app.route('/threshold/<int:idx>')
def threshold(idx):
    r = Container.handler['threshold%d' % idx](request.args)
    return r


# @app.route('/click/<ele_id>')
# def click(ele_id):
#     r = Container.click_handler[ele_id](request.args)
#     if not isinstance(r, str):
#         r = jsonify(r)
#     return r


# @app.route('/brush/<ele_id>')
# def brush(ele_id):
#     r = Container.brush_handler[ele_id](request.args)
#     if not isinstance(r, str):
#         r = jsonify(r)
#     return r


@app.route('/ite/')
def input_text():
    r = Container.brush_handler['ite'](request.args)
    return r


# @app.route('/gtl/te')
# def gallery_sentence():
#     r = Container.click_handler['gtl'](request.args)
#     return r
#
#
# @app.route('/select_sen/')
# def select_sentence():
#     r = Container.other_handler['select_sen'](request.args)
#     return r
#
#
# @app.route('/select_layer/')
# def select_layer():
#     r = Container.other_handler['select_layer'](request.args)
#     return r
#
#
# @app.route('/select_i/<ele_id>')
# def select_i(ele_id):
#     r = Container.si[ele_id](request.args)
#     return r


@app.route('/set_click_mode/<int:mode>')
def set_click_mode(mode):
    Container.other_handler['set_set_mode'](mode)
    return jsonify("Success")


@app.route('/set_set_mode/<int:mode>')
def set_set_mode(mode):
    Container.other_handler['set_set_mode'](mode)
    return jsonify("Success")


@app.route('/msv/<int:gid>')
def msv(gid):
    r = Container.other_handler['g%d' % gid]()
    return r


@app.route('/which_selector/<int:gid>')
def which_selector(gid):
    r = Container.other_handler['ws%d' % gid]()
    return r


@app.route('/switch_selector/<int:gid>')
def switch_selector(gid):
    r = Container.other_handler['ss%d' % gid](request.args)
    return r


@app.route('/set_code/<int:gid>')
def set_code(gid):
    Container.other_handler['sc%d' % gid](request.args)
    return jsonify('Success')


@app.route('/add_selector/<int:gid>')
def add_selector(gid):
    Container.other_handler['as%d' % gid](request.args)
    return jsonify('Success')


# @app.route('/t1/')
# def t1():
#     r = Container.other_handler['t1'](request.args)
#     return r


@app.route('/reset_prev_r/')
def reset_prev_r():
    Container.other_handler['init_prev_r']()
    return jsonify('Success')


# @app.route('/t2/')
# def t2():
#     r = Container.other_handler['t2']()
#     return r
def open_browser():
    webbrowser.open_new_tab('http://localhost:5000')


def launch(auto_open=False):
    if auto_open:
        threading.Timer(1.25, open_browser).start()
    # Don't set debug to True
    app.run(host='0.0.0.0', debug=False)
