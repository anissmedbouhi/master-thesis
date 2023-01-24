import shutil
import numpy as np
import pandas as pd
import torch
import datasets
from imp import reload
from tqdm import tqdm
from matplotlib import pyplot as plt
plt.style.use('ggplot')

from sklearn import mixture
from scipy import stats

import umap
import sklearn
UMAP = umap.UMAP
ISOMAP = sklearn.manifold.Isomap
TSNE = sklearn.manifold.TSNE

import models
from models import save_model, load_model
from losses import *

from plots import *
from utils import split, cuda, isomap_kernel

umap_ = umap.umap_
from algebraic import find_cut_alphas, create_simplicial_complex, flatten_complex, plot_graph

from torch.utils.data import TensorDataset, DataLoader

from vaes import *

from dijkstra import *