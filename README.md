## *SplineCam*: Exact Visualization and Characterization of Deep Network Geometry and Decision Boundaries
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://bit.ly/splinecam-demo)

### [Paper Link](#) | [Website](https://bit.ly/splinecam)

#### *Authors:* Ahmed Imtiaz Humayun<sup>1</sup>, Randall Balestriero<sup>2</sup>, Guha Balakrishnan<sup>1</sup>, Richard Baraniuk<sup>1</sup>
<sup>1</sup>Rice University, <sup>2</sup>Meta AI, FAIR

**Abstract**: Current Deep Network (DN) visualization and interpretability methods rely heavily on data space visualizations such as scoring which dimensions of the data are responsible for their associated prediction or generating new data features or samples that best match a given DN unit or representation. In this paper, we go one step further by developing the first provably exact method for computing the geometry of a DN's mapping -- including its decision boundary -- over a specified region of the data space. By leveraging the theory of Continuous Piece-Wise Linear (CPWL) spline DNs, \textbf{SplineCam} exactly computes a DN's geometry without resorting to approximations such as sampling or architecture simplification. SplineCam applies to any DN architecture based on CPWL nonlinearities, including (leaky-)ReLU, absolute value, maxout, and max-pooling and can also be applied to regression DNs such as implicit neural representations. Beyond decision boundary visualization and characterization, SplineCam enables one to compare architectures, measure generalizability and sample from the decision boundary on or off the manifold.
