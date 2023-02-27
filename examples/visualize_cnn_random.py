import sys
import os

sys.path.append('..')
import splinecam as sc

import torch
import matplotlib.pyplot as plt

model = torch.nn.Sequential(
    *[
        torch.nn.Conv2d(3, 6, 5, stride=2, padding=2, bias=False),
        torch.nn.BatchNorm2d(6),
        torch.nn.ReLU(),
        torch.nn.Conv2d(6, 16, 5, stride=2, padding=2, bias=False),
        torch.nn.ReLU(),
        torch.nn.Flatten(),
        torch.nn.Linear(16 * 8 * 8, 120),
        torch.nn.Linear(120, 84),
        torch.nn.Linear(84, 10)
    ]
)

model.eval()
model.type(torch.float64)

NN = sc.wrappers.model_wrapper(
    model,
    input_shape=(3,32,32),
    T = torch.randn(3*32*32,2),
    dtype = torch.float64
)

print('forward and affine equivalency flag ', NN.verify())

global_dtype = torch.float64

poly = sc.utils.create_polytope_2d(scale=1,seed=10)+2
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
    
sc.plot.plot_partition(out_cyc, xlims=[poly[:,0].cpu().min().numpy(),
                                         poly[:,0].cpu().max().numpy()],
                         ylims=[poly[:,1].cpu().min().numpy(),poly[:,1].cpu().max().numpy()])

plt.savefig('../figures/cnn_visualize.jpg',transparent=True, bbox_inches=0, pad_inches=0)