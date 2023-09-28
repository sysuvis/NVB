[Beta Version](https://github.com/sysuvis/NVB-beta)

# Getting started with NNVisBuilder

To use NNVisBuilder, first run `mlp.py` and see how it works. If you run `cnn.py` or `kdt.py`, the relevant dataset will be automatically downloaded first. If you run `lstm.py`, you need to manually download `eng-fra.txt` from the data folder at [NNVisBuilder Big Files](https://pan.baidu.com/s/1IQKDWLq65XKhWv1ObXAoxQ?pwd=1234) and place it in the data folder, and download `encoder.pth` and `decoder.pth` from the model folder and place them in the model folder.

(When selecting a dataset in the `mlp.py` interface, you need to click on the circle on the left side to trigger, and it will be optimized in the future.)

(In the interface of `cnn.py`, to brush the scatter plot, you need to first click the "click" button in the upper left corner and change the state to "brush" mode.)

(The version requirements in requirements.txt can actually be lower, for example, the PyTorch version can be 1.2.0, but the specific version requirements still need to be tested.)

# What is NNVisBuilder

NNVisBuilder is a programming toolkit designed to enable users to easily and quickly build interactive visual analytic interfaces for various neural networks. While we provide some interface templates, they are not a core part of NNVisBuilder. NNVisBuilder is a visualization system that helps users build their own interfaces.

# How to Code with NNVisBuilder to Build Your Own Interface

NNVisBuilder is like a visual analytics framework, but its visualization is based on views. To add a view to the interface, simply create an object for that type of view. For example:

`view = ScatterPlot(data, [100, 100], [200, 200])`

Generally, views need to be bound to data, which specifies how elements are displayed within the view. For instance, a scatter plot consists of a collection of points, and its first parameter specifies the positions of those points. In addition to positions, the ScatterPlot can also specify colors, sizes, etc., all of which can be defined through data.

Furthermore, users can specify additional view information such as position and size. For example, the above statement specifies a position of `[100, 100]` and a size of `[200, 200]`.

The binding between views and data in NNVisBuilder is dynamic. As data changes, views change as well, providing the foundation for NNVisBuilder's interactive functionality.

If you want to implement some interactive functionality in NNVisBuilder, you don't need to write complex code to modify the interface. Simply modify the data, and the associated views will update accordingly.

NNVisBuilder's views require data to be wrapped in a Data class to achieve dynamic binding. For instance, if `a` is a numpy array representing all the points' positions, wrap it using:

`data = Data(a)`

and then pass it to the view. Not all views require wrapped data, and users may choose not to wrap data if they don't need dynamic relationships.

## Interaction

Defining interactive methods is similar to JavaScript:

`view.onclick(f)`

The f here is the event handler function of the following form:

```def f(value):
   data.update(value)
```
What `f`'s argument takes depends on the specific view. There will be some commonality for ease of use and some flexibility for specialized designs.

Specifically, `f` allows for different numbers of arguments to be passed. For example, in a HeatMap, if the Selector is set to select one or more rows (i.e., selecting on the first dimension of the tensor), then value will be the row number or a list of row numbers. If the Selector is set to select specific points (such as the point at the first row and second column <1,2>) or a list of points, then the definition of f is a function that accepts two arguments, and these two arguments will represent the x and y coordinates of the selected point. Sometimes, for the same Selector, `f` can also be defined to accept different numbers of arguments, depending on the specific view. These details will be reflected in the user manual later.

In `f`, data or some transformations can be modified. If the types of value and data are the same (the tensor order is the same), calling update will directly replace the value with the corresponding value in value. Otherwise, if data is a vector and value is a scalar, if value is in data, it will be removed, otherwise it will be added. This is a convenience we provide based on practice.

Views can define a Highlighter to specify how to respond to selected information. For example, for a scatter plot, it can be specified to enlarge the selected point or modify the color of the point. NNVisBuilder includes the selected information in the Highlighter of the view, so if two views are bound to the same Highlighter, their selections will be passed on.

NNVisBuilder defines default event handling functions for views. This function will use the value passed to f to modify the selection information of the view's Highlighter. At the same time, NNVisBuilder allows mapping functions to be added for selected information. When the selected information changes, all corresponding mapping functions will be called automatically. For example, the following code:
```
def f(value):
    highlighter = view.highlighter
    highlighter.update(value)
    g(highlighter.value)
view.on_click(f)
```
And the code:
```
view.highlighter.add_mapping(g)
```
have the same effect.

## Structure

Therefore, data, views, and interactions are the three basic modules of NNVisBuilder. To use NNVisBuilder, users need to prepare data, define views, and specify interactions.

NNVisBuilder has some designs in three modules, and these designs based on our data abstraction are our main contributions, or the main differences that make NNVisBuilder different from other visualization frameworks (in building neural network visual analysis interfaces)：

1. Data:
   - NNVisBuilder encapsulates the common process of data acquisition and abstracts the data as data classes. The binding relationship between data and views can be achieved based on these data classes.
   - NNVisBuilder provides common transformations on data. The transformations using NNVisBuilder will automatically record the transformation relationship (forming a transformation relationship graph), and other data will also change accordingly when some data change.
   - Dynamic binding and transformation relationship graph provide a foundation for users to interact at a higher level.
2. Views:
   - Users can specify different attributes of the views when defining them, including some common attributes such as view position and size, and some specific attributes of the views, such as the position and color of each point in a scatter plot.
   - The specific attributes of the views can be specified as an NNVisBuilder data, which can achieve dynamic binding. If dynamic binding is not needed, the value of the attribute can be directly specified, and there is no need to wrap the value as an NNVisBuilder data.
   - Users only need to specify the attributes of the views, and the system will automatically generate the visualization. Users do not need to care about the visualization aspect or write visualization code.
   - If users need to customize the view, they can define their own view based on NNVisBuilder specifications (currently referring to existing view classes), and then use it like using an existing view.
   - We will provide richer preset view types in the future.
   - NNVisBuilder also provides some widgets to help users build control panels, which are generally similar to views, but do not require binding data.
3. Interaction:
   - Based on dynamic binding, users can achieve all types of interaction by modifying the data, transformations or selection information corresponding to other views in the event handling of the interaction. This is **the most fundamental difference in using NNVisBuilder for coding**, based on our data abstraction.
   - Selection information is another factor that affects the display result of the views, usually represented as a selected subset of a certain dimension of the data. For example, a heatmap corresponds to a two-dimensional tensor data, and one row (multiple rows) or one column (multiple columns) can be selected.
   - Selection information may undergo some transformations when it affects the data or selection information of other views. NNVisBuilder provides templates for common transformations to further facilitate user coding.
   - NNVisBuilder defines different selectors for different selection methods, and each type of view has multiple preset selectors, and users can also customize selectors.
   - NNVisBuilder abstracts the response of highlighted information in the view as Highlighter, and each type of view has multiple preset highlighters, and users can also customize highlighters.
   - NNVisBuilder provides Multi-Selector and Multi-Highlighter, which allow users to obtain more selection methods and interaction response modes through combination.
   - The specific description of Selector, Highlighter, Multi-Selector, and Multi-Highlighter is to be supplemented.

As a toolkit specialized for neural networks, NNVisBuilder is designed to:

1. Abstract the interface representation model, which summarizes the data processing and interaction processes of the interface as a flow chart
2. Encapsulate the process of obtaining commonly used data for neural network visualization, such as network activation, gradients, and connections

## Some additional explanations:

1. Composite view: By overlapping some existing views or aligning some views closely to each other (NNVisBuilder provides such alignment functionality), composite views can be created.
2. Multiple models: If data from multiple models is needed, simply create multiple builders. Finally, calling the run method of one of the builders can generate the interface.
3. Other transformations: Other transformations like TSNE can also be added to the relationship tree (participating in dynamic response after modifying the data) by using `data.apply_transform(OtherTransform(tsne))`. If the TSNE transformation does not need to record relationships, it can be used directly with `tsne(data)`.

Further explanations, detailed instructions, user manuals, and API documents will be provided in the future.

Please refer to [this manual](https://github.com/sysuvis/NVB/blob/main/documents/manual.md) for more information on implementation and API.
