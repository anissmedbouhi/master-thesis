# Information
This repository contains the code for my master thesis called "Towards topology-aware Variational Auto-Encoders: from InvMap-VAE to Witness Simplicial VAE" and for the related article entitled ["InvMap and Witness Simplicial Variational Auto-Encoders"](https://www.mdpi.com/2504-4990/5/1/14) published in MDPI Machine Learning And Knowledge Extraction journal under the topic Topology vs. Geometry in Data Analysis/Machine Learning. <br />
<br />
The work was performed at: Division of Robotics Perception and Learning, Department of Intelligent Systems, School of Electrical Engineering and Computer Science, KTH Royal Institute of Technology. <br />
<br />
Main supervisor of the master thesis: Dr. Anastasiia Varava <br />
Co-supervisor: Vladislav Polianskii (PhD student) <br />
Examiner: Prof. Danica Kragic Jensfelt <br />
<br />
The thesis is publicly available on KTH Publication Database DiVA: [permanent link here](http://urn.kb.se/resolve?urn=urn:nbn:se:kth:diva-309487) or if the link does not work try [here](http://kth.diva-portal.org/smash/record.jsf?pid=diva2%3A1642320&dswid=-7436).

# To cite the related article

```
@Article{make5010014,
AUTHOR = {Medbouhi, Aniss Aiman and Polianskii, Vladislav and Varava, Anastasia and Kragic, Danica},
TITLE = {InvMap and Witness Simplicial Variational Auto-Encoders},
JOURNAL = {Machine Learning and Knowledge Extraction},
VOLUME = {5},
YEAR = {2023},
NUMBER = {1},
PAGES = {199--236},
URL = {https://www.mdpi.com/2504-4990/5/1/14},
ISSN = {2504-4990},
ABSTRACT = {Variational auto-encoders (VAEs) are deep generative models used for unsupervised learning, however their standard version is not topology-aware in practice since the data topology may not be taken into consideration. In this paper, we propose two different approaches with the aim to preserve the topological structure between the input space and the latent representation of a VAE. Firstly, we introduce InvMap-VAE as a way to turn any dimensionality reduction technique, given an embedding it produces, into a generative model within a VAE framework providing an inverse mapping into original space. Secondly, we propose the Witness Simplicial VAE as an extension of the simplicial auto-encoder to the variational setup using a witness complex for computing the simplicial regularization, and we motivate this method theoretically using tools from algebraic topology. The Witness Simplicial VAE is independent of any dimensionality reduction technique and together with its extension, Isolandmarks Witness Simplicial VAE, preserves the persistent Betti numbers of a dataset better than a standard VAE.},
DOI = {10.3390/make5010014}
}
```

# Abstract of the master thesis
Variational Auto-Encoders (VAEs) are one of the most famous deep generative models. After showing that standard VAEs may not preserve the topology, that is the shape of the data, between the input and the latent space, we tried to modify them so that the topology is preserved. This would help in particular for performing interpolations in the latent space.<br />
Our main contribution is two folds. Firstly, we propose successfully the InvMap-VAE which is a simple way to turn any dimensionality reduction technique, given its embedding, into a generative model within a VAE framework providing an inverse mapping, with all the advantages that this implies. Secondly, we propose the Witness Simplicial VAE as an extension of the Simplicial Auto-Encoder to the variational setup using a Witness Complex for computing a simplicial regularization. The Witness Simplicial VAE is independent of any dimensionality reduction technique and seems to better preserve the persistent Betti numbers of a data set than a standard VAE, although it would still need some further improvements.<br />
Finally, the two first chapters of this master thesis can also be used as an introduction to Topological Data Analysis, General Topology and Computational Topology (or Algorithmic Topology), for any machine learning student, engineer or researcher interested in these areas with no background in topology.

# Keywords
Variational Auto-Encoder, Nonlinear dimensionality reduction, Generative model, Inverse projection, Computational topology, Algorithmic topology, Topological Data Analysis, Data visualisation, Unsupervised representation learning, Topological machine learning, Betti number, Simplicial complex, Witness complex, Simplicial map, Simplicial regularization.

# Help
All you need to run is in the main file.

To run InvMap-VAE you need to run:
- Directory, packages, and fix seeds
- Define the dataset, train = (matrix X, colors)
- Dimensionality reduction (to get the embedding you want to use)
- Artificial Neural Networks: Model, and then InvMap-VAE
- Results: Loss, latent space and reconstruction visualizations

To run Witness Simplicial VAE you need to run:
- Directory, packages, and fix seeds
- Define the dataset, train = (matrix X, colors)
- Build the Witness Complex directly from the input data space
- Artificial Neural Networks: Model, and then Witness Simplicial VAE
- Results: Loss, latent space and reconstruction visualizations

To run Isolandmarks Witness Simplicial VAE you need to run:
- Directory, packages, and fix seeds
- Define the dataset, train = (matrix X, colors)
- Build the Witness Complex directly from the input data space
- Build approximative geodesics distance matrix of the landmarks points given a witness complex
- Artificial Neural Networks: Model, and then Isolandmarks Witness Simplicial VAE
- Results: Loss, latent space and reconstruction visualizations

# Acknowledgments

This project was made possible thanks to my supervisors and colleagues at RPL. We are very thankful to Giovanni Lucas Marchetti for all the helpful and inspiring discussions, and his valuable feedback. I am also very thankful to Achraf Bzili for his help with technical implementation at the time of the master thesis.

Sources directly used for the implementations of our methods:
- https://github.com/MrBellamonte/AEs-VAEs-TDA/blob/master/src/topology/witness_complex.py thank you Simon!
- https://github.com/crixodia/python-dijkstra/blob/master/dijkstra.py gracias Cristian Bastidas!

Sources used as inspiration to start with the project:
- https://github.com/BorgwardtLab/topological-autoencoders
- https://github.com/jgalle29/simplicial from Jose Gallego-Posada.
