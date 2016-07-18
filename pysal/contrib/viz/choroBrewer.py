

get_ipython().magic(u'matplotlib inline')
from ipywidgets import Dropdown, RadioButtons, IntSlider, VBox, FloatSlider, Button, interact
from ipywidgets import HBox
from IPython.display import display
from pysal.contrib.viz.color import color_display_types
from pysal.contrib.viz.color import plot_cmaps as pcmaps
from pysal.contrib.viz import mapping as maps
import pysal as ps



def choro(gd, col):
    dt = sorted(color_display_types.keys(), reverse=True)
    data_type = RadioButtons(description='Data Type', options=dt)
    bindings = {}
    for t in dt:
        bindings[t] = color_display_types[t].keys()

    cmap_dd = Dropdown(description='CMap:', options=bindings[data_type.value])

    def type_change(change):
        with cmap_dd.hold_trait_notifications():
            cmap_dd.options = bindings[change['new']]
            k_dd.options = kbindings[change['new']]
            #plot_cmaps('diverging')

    def cmap_change(change):
            with cmap_dd.hold_trait_notifications():
                print('new cmap', str(change['new']))

    data_type.observe(type_change, names=['value'])
    cmap_dd.observe(cmap_change, names=['value'])
    kbindings = {'sequential': range(3,9+1),
                'qualitative': range(3,12+1),
                'diverging': range(3,11+1)}
    k_dd = Dropdown(description='k', options=kbindings[data_type.value])

    display(HBox([data_type, k_dd]))
    @interact(cmap=cmap_dd)
    def plot_cmaps(cmap):
        i = cmap_dd.options.index(cmap)
        pcmaps(data_type.value, i)
        maps.geoplot(gd, col, k=int(k_dd.value),
                    palette=cmap_dd.value, dtype=data_type.value,
                    figsize=(6, 6))
