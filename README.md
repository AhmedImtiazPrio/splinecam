## *SplineCam*: Exact Visualization and Characterization of Deep Network Geometry and Decision Boundaries, CVPR 2023
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://bit.ly/splinecam-demo)

### [Paper Link](https://arxiv.org/abs/2302.12828) | [Website](https://bit.ly/splinecam)
<img src="https://user-images.githubusercontent.com/32792313/221405026-ba0e1d12-5a25-4937-9cdc-b97a84dd1d8f.jpg" width="100%">

Fig: Learned implicit surface by a Neural SDF with annotations for three 2D slices (Left). SplineCam visualizations for annotated 2D slices (Rest) with decision boundary (or zero level set of the output).

#### *Authors:* Ahmed Imtiaz Humayun<sup>1</sup>, Randall Balestriero<sup>2</sup>, Guha Balakrishnan<sup>1</sup>, Richard Baraniuk<sup>1</sup>
<sup>1</sup>Rice University, <sup>2</sup>Meta AI, FAIR

**Abstract**: Current Deep Network (DN) visualization and interpretability methods rely heavily on data space visualizations such as scoring which dimensions of the data are responsible for their associated prediction or generating new data features or samples that best match a given DN unit or representation. In this paper, we go one step further by developing the first provably exact method for computing the geometry of a DN's mapping -- including its decision boundary -- over a specified region of the data space. By leveraging the theory of Continuous Piece-Wise Linear (CPWL) spline DNs, \textbf{SplineCam} exactly computes a DN's geometry without resorting to approximations such as sampling or architecture simplification. SplineCam applies to any DN architecture based on CPWL nonlinearities, including (leaky-)ReLU, absolute value, maxout, and max-pooling and can also be applied to regression DNs such as implicit neural representations. Beyond decision boundary visualization and characterization, SplineCam enables one to compare architectures, measure generalizability and sample from the decision boundary on or off the manifold.

https://user-images.githubusercontent.com/32792313/221407228-86f1a36b-1d88-43f6-abba-1554049c842c.mp4

**Video**: SplineCam visualizations during training of a binary classifier MLP with width 10 and depth 5. Regions are colored by the norm of their corresponding slope parameters. Notice how the function keeps changing even when the decision boundary has converged, especially away from the training data. 

## Examples

Examples are placed under the `./example` folder. Google colabs are also provided for some.

| Model | Data | Filename | Link
| :---- | :---- | :---- | :----
| MLP | Two Moons, Two Circles, Two Blobs | toy2d.py |  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://bit.ly/splinecam-demo)
| MLP | f : 2 &rarr; 1 | -- |  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1U9Ql5Ww3dJdgTTQvXOKqVgflWTr1kD3s)
| MLP - Implicit Neural Representation | 2D image | 2d_inr.ipynb | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://bit.ly/splinecam-demo-anon)
| Scattering Network | 1D Signal/Audio | scattering.ipynb | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://bit.ly/scattering-cam)
| MLP | randomly initialized | visualize_mlp_random.py | [Link](https://github.com/AhmedImtiazPrio/splinecam/blob/main/examples/visualize_mlp_random.py)
| CNN | randomly initialized | visualize_cnn_random.py | [Link](https://github.com/AhmedImtiazPrio/splinecam/blob/main/examples/visualize_cnn_random.py)
| VGG11 | tinyimagenet-200 | - |  


## Usage

To begin, first we have to define a 2D input space region of interest that we wish to visualize. The region of interest can be a polytopal region **P** at the input of the network defined via a set of vertices. Since all the region finding operations are performed in 2D, we also require a projection operator that projects vectors from the input space onto **P**. The following shows how to wrap any torch sequential for visualization

```python
import torch
import splinecam

## given torch network `model` and region of interest `domain`

T = splinecam.utils.get_proj_mat(domain)

model.cuda()
model.eval()
model.type(torch.float64)

print('Wrapping model with SplineCam...')
NN = splinecam.wrappers.model_wrapper(
    model,
    input_shape=model.input_shape,
    T = T,
    dtype = torch.float64,
    device = 'cuda'
)

## check .forward() and matmul operation equivalence
print('Verifying wrapped model...')
flag =  NN.verify()
print('Model.forward and matmul equivalence check', flag)
assert flag

```
Compute the partitions on **P** along with the decision boundary and plot
```python
regions, db_edges = splinecam.compute.get_partitions_with_db(domain,T,NN)

fig,ax = plt.subplots()
splinecam.plot.plot_partition(regions,
                              alpha=0.3, edgecolor='#a70000',ax=ax,
                              color_range=[.3,.8], linewidth=.5)
for each in db_edges:
    ax.plot(each[:,0], each[:,1], c='r', linewidth=5)
```

## Requirements

SplineCam is mostly implemented using Pytorch and Graph-tool. All the linear algebra operations are performed using Pytorch and are vectorized, therefore scalable using GPUs. The Graph-tool operations are single threaded.

```
torch>=1.9,<=1.12
tqdm
networkx
python-igraph>=0.10
graph-tool==2.45
livelossplot ## to keep track of training
```

## Setup

Clone repo
```bash
git clone https://github.com/AhmedImtiazPrio/splinecam.git
```

Setup graph-tool and pycairo
```bash
echo "deb http://downloads.skewed.de/apt jammy main" >> /etc/apt/sources.list
apt-key adv --keyserver keyserver.ubuntu.com --recv-key 612DEFB798507F25
apt-get update
apt-get install python3-graph-tool python3-matplotlib python3-cairo

apt purge python3-cairo
apt install libcairo2-dev pkg-config python3-dev
pip install --force-reinstall pycairo
pip install zstandard
```

Install misc network libraries (will be removed in future release)
```bash
pip install networkx
pip install igraph>=0.10
```

Install misc dependencies
```bash
pip install tqdm
pip install zstandard
pip install livelossplot
pip install imageio-ffmpeg
```
If PyTorch is not installed
```bash
pip install --pre torch==1.12+cu116 torchvision -f https://download.pytorch.org/whl/torch_stable.html
```

## Release Notes

The SplineCam python toolbox can wrap any given Pytorch sequential network, containing a set of supported modules. While the number of modules currently supported is not exhaustive, we will be adding support for newer Pytorch modules over time.

Currently supported Pytorch modules are:

```python
torch.nn.modules.linear.Linear,
torch.nn.modules.Sequential, ##also used to skip layers
torch.nn.modules.BatchNorm1d,
torch.nn.modules.Conv2d,
torch.nn.modules.BatchNorm2d,
torch.nn.modules.Flatten,
torch.nn.modules.AvgPool2d,
torch.nn.modules.Dropout
```

SplineCam is theoretically exact for any piecewise linear activation function, e.g., LeakyReLU, Sawtooth. Currently supported activation functions are:

```python
torch.nn.modules.activation.ReLU,
torch.nn.modules.activation.LeakyReLU,
```

## To do

1. Convert all igraph to graph-tool
2. Add per layer visualization support. Save meta for layer index in graph for each edge.
3. Add support for periodic activation functions
4. Add support for skip connections

## Citation
```

@inproceedings{humayun2022splinecam,
  title={SplineCam: Exact Visualization and Characterization of Deep Network Geometry and Decision Boundaries},
  author={Humayun, Ahmed Imtiaz and Balestriero, Randall and Balakrishnan, Guha and Baraniuk, Richard},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023}
}


@inproceedings{
humayun2022exact,
title={Exact Visualization of Deep Neural Network Geometry and Decision Boundary},
author={Ahmed Imtiaz Humayun and Randall Balestriero and Richard Baraniuk},
booktitle={NeurIPS 2022 Workshop on Symmetry and Geometry in Neural Representations},
year={2022},
}
```
