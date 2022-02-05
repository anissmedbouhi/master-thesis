# Information
This repository contains the code for my master thesis called "Towards topology-aware Variational Auto-Encoders: from InvMap-VAE to Witness Simplicial VAE". <br />
<br />
It was performed at: Division of Robotics Perception and Learning, Department of Intelligent Systems, School of Electrical Engineering and Computer Science, KTH Royal Institute of Technology. <br />
<br />
Main supervisor: Dr. Anastasiia Varava <br />
Co-supervisor: Vladislav Polianskii (PhD student) <br />
Examiner: Prof. Danica Kragic Jensfelt <br />
<br />
The thesis will be publicly available on KTH website.

# Abstract of the master thesis

Variational Auto-Encoders (VAEs) are one of the most famous deep generative models. After showing that standard \acrshortpl{VAE} may not preserve the topology, that is the shape of the data, between the input and the latent space, we tried to modify them so that the topology is preserved. This would help in particular for performing interpolations in the latent space.<br />
Our main contribution is two folds. Firstly, we propose successfully the InvMap-VAE which is a simple way to turn any dimensionality reduction technique, given its embedding, into a generative model within a VAE framework providing an inverse mapping, with all the advantages that this implies. Secondly, we propose the Witness Simplicial VAE as an extension of the Simplicial Auto-Encoder to the variational setup using a Witness Complex for computing a simplicial regularization. The Witness Simplicial VAE is independent of any dimensionality reduction technique and seems to better preserve the persistent Betti numbers of a data set than a standard VAE, although it would still need some further improvements.<br />
Finally, the two first chapters of this master thesis can also be used as an introduction to Topological Data Analysis, General Topology and Computational Topology (or Algorithmic Topology), for any machine learning student, engineer or researcher interested in these areas with no background in topology.

# Keywords
Variational Auto-Encoder, Nonlinear dimensionality reduction, Generative model, Inverse projection, Computational topology, Topological Data Analysis, Data visualisation, Unsupervised representation learning, Topological machine learning, Betti numbers, Simplicial map, Simplicial regularization.
