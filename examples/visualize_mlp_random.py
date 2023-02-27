#!/usr/bin/env python
# coding: utf-8

import sys
import os
sys.path.append('..')

import splinecam as sc
import torch
import matplotlib.pyplot as plt


in_shape = 10
out_shape = 1
width = 20
depth = 10
# act = torch.nn.ReLU(inplace=True)
act = torch.nn.LeakyReLU(0.03)

layers = []

offset = 1

layers.append(torch.nn.Linear(in_shape,width+offset))
layers.append(act)


for i in range(depth-1):
    layers.append(torch.nn.Linear(width+offset,width+offset+1))
    layers.append(torch.nn.BatchNorm1d(width+offset+1))
    layers.append(act)
    offset += 1
    
layers.append(torch.nn.Linear(width+offset,out_shape))


model = torch.nn.Sequential(*layers)
model.cuda()


## wrap model 
global_dtype = torch.float64

model.eval().cuda()
model.type(global_dtype)
NN = sc.wrappers.model_wrapper(model, T=torch.randn(10,2), dtype=global_dtype)
print('forward and affine equivalency flag', NN.verify())

## input region
poly = sc.utils.create_polytope_2d(scale=1,seed=10)
poly = torch.from_numpy(poly).cuda().to(global_dtype)

Abw = NN.layers[0].get_weights()[None,...]
out_cyc = [poly]

for current_layer in range(1,len(NN.layers)):

    out_cyc,out_idx = sc.graph.to_next_layer_partition(
        cycles = out_cyc,
        Abw = Abw,
        NN = NN,
        current_layer = current_layer,
        dtype = global_dtype
    )
    
    with torch.no_grad():

        means = sc.utils.get_region_means(out_cyc, dims=out_cyc[0].shape[-1], dtype=global_dtype)
        means = NN.layers[:current_layer].forward(means.cuda())

        Abw = sc.utils.get_Abw(
            q = NN.layers[current_layer].get_activation_pattern(means),
            Wb = NN.layers[current_layer].get_weights(),
            incoming_Abw = Abw[out_idx]
                )
    
sc.plot.plot_partition(out_cyc, xlims=[-.8,.8], ylims=[-.8,.8], color_range=[.4,.8])
plt.savefig('../figures/mlp_visualize.jpg',transparent=True, bbox_inches=0, pad_inches=0)
