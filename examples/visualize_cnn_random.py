import sys
import os

sys.path.append('..')
import splinecam as sc

import torch
import matplotlib.pyplot as plt

### define model

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

### define input domain to compute partitions

# prescribe input domain for partition computation
domain = sc.utils.get_square_slice_from_one_anchor(torch.randn(1,3*32*32),
                                                   pad_dist=2,
                                                   seed=None)

# compute linear projection from input space to target domain
T = sc.utils.get_proj_mat(domain)

# wrap model with splinecam library
NN = sc.wrappers.model_wrapper(
    model,
    input_shape=(3,32,32),
    T = T,
    dtype = torch.float64
)

print('forward and affine equivalency flag ', NN.verify())

### Compute regions and decision boundary

out_cyc,endpoints,Abw = sc.compute.get_partitions_with_db(domain,T,NN)

### Plot partitions

minval,_ = torch.vstack(out_cyc).min(0)
maxval,_ = torch.vstack(out_cyc).max(0)

sc.plot.plot_partition(out_cyc, xlims=[minval[0],maxval[0]],alpha=0.3,
                         edgecolor='#a70000',color_range=[.3,.8],
                         colors=['#469597', '#5BA199', '#BBC6C8', '#E5E3E4', '#DDBEAA'],
                         ylims=[minval[1],maxval[1]], linewidth=.5)

plt.savefig('../figures/cnn_visualize.jpg',transparent=True, bbox_inches=0, pad_inches=0)