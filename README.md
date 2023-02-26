## *SplineCam*: Exact Visualization and Characterization of Deep Network Geometry and Decision Boundaries
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://bit.ly/splinecam-demo)

### [Paper Link](#) | [Website](https://bit.ly/splinecam)
<img src="https://user-images.githubusercontent.com/32792313/221405026-ba0e1d12-5a25-4937-9cdc-b97a84dd1d8f.jpg" height="250">

Fig: Learned implicit surface by a Neural SDF with annotations for three 2D slices (Left). SplineCam visualizations for annotated 2D slices (Rest)

#### *Authors:* Ahmed Imtiaz Humayun<sup>1</sup>, Randall Balestriero<sup>2</sup>, Guha Balakrishnan<sup>1</sup>, Richard Baraniuk<sup>1</sup>
<sup>1</sup>Rice University, <sup>2</sup>Meta AI, FAIR

**Abstract**: Current Deep Network (DN) visualization and interpretability methods rely heavily on data space visualizations such as scoring which dimensions of the data are responsible for their associated prediction or generating new data features or samples that best match a given DN unit or representation. In this paper, we go one step further by developing the first provably exact method for computing the geometry of a DN's mapping -- including its decision boundary -- over a specified region of the data space. By leveraging the theory of Continuous Piece-Wise Linear (CPWL) spline DNs, \textbf{SplineCam} exactly computes a DN's geometry without resorting to approximations such as sampling or architecture simplification. SplineCam applies to any DN architecture based on CPWL nonlinearities, including (leaky-)ReLU, absolute value, maxout, and max-pooling and can also be applied to regression DNs such as implicit neural representations. Beyond decision boundary visualization and characterization, SplineCam enables one to compare architectures, measure generalizability and sample from the decision boundary on or off the manifold.

https://user-images.githubusercontent.com/32792313/221407228-86f1a36b-1d88-43f6-abba-1554049c842c.mp4

**Video**: SplineCam visualizations during training of a binary classifier MLP with width 10 and depth 5. Regions are colored by the norm of their corresponding slope parameters. Notice how the function keeps changing even when the decision boundary has converged, especially away from the training data. 

### Citation
```
@inproceedings{
humayun2022exact,
title={Exact Visualization of Deep Neural Network Geometry and Decision Boundary},
author={Ahmed Imtiaz Humayun and Randall Balestriero and Richard Baraniuk},
booktitle={NeurIPS 2022 Workshop on Symmetry and Geometry in Neural Representations},
year={2022},
}
```
