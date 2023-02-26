import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as mpl_Polygon
from matplotlib.collections import PatchCollection
import matplotlib.colors as mpl_colors
import mpl_toolkits.mplot3d as a3


import numpy as np
from networkx import draw as nxdraw
from graph_tool.draw import graph_draw

import tqdm

def plot_partition(cycles,ax=None,xlims=[-2,2],ylims=[-2,2],
                   edgecolor='w',
                   linewidth=.3,
                   alpha=1,
                   colors=None,figsize=(15,15),
                   color_range=[.3,1]):
    '''
    Plot all cycles
    '''
    
    if ax is None:
        fig, ax = plt.subplots(1,figsize=figsize)

    if colors is None:
    
        for cyc in tqdm.tqdm(cycles,total=len(cycles)):
                polygon = mpl_Polygon(cyc, True, facecolor=mpl_colors.rgb2hex(
                    np.clip(np.random.rand(3),color_range[0],color_range[1])
                ))
                polygon.set_edgecolor(edgecolor)
                polygon.set_linewidth(linewidth)
                polygon.set_alpha(alpha)
                ax.add_patch(polygon)
                
    else:
        
        for cyc in tqdm.tqdm(cycles,total=len(cycles)):
                polygon = mpl_Polygon(cyc, True, facecolor=colors[np.random.randint(0,len(colors))])
                polygon.set_edgecolor(edgecolor)
                polygon.set_linewidth(linewidth)
                polygon.set_alpha(alpha)
                ax.add_patch(polygon)
        
            
    ax.set_xticks([])
    ax.set_yticks([])
    
    if xlims is not None:
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        
        
def plot_networkx_graph(G):
    pos = dict(zip(G.nodes,[G.nodes[each]['v'].numpy() for each in G.nodes]))
    nxdraw(G,pos=pos,with_labels=True,node_size=1,font_size=5)
    
    
def plot_graphtool_graph(G):
    graph_draw(G,pos=G.vp['v'])