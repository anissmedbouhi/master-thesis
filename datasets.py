from abc import ABCMeta, abstractmethod
from typing import Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets

class DataSet(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def sample(self, n_samples: int, seed: int, noise: float, train: bool) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @property
    @abstractmethod
    def fancy_name(self) -> str:
        pass


DEFAULT = {
    "cylinder" : dict(seed=1, horizontal = False, center = [0,0,0], height = 10, width = 1, radius = 4, noise = 0),
    "swissroll" : dict(seed = 1, noise = 0)
}

class Cylinder(DataSet):
    '''
    Cylinder
    '''
    fancy_name = "Cylinder dataset"

    __slots__ = []
    def __init__(self):
        pass

    def sample(self, n_samples, horizontal=DEFAULT['cylinder']['horizontal'], center=DEFAULT['cylinder']['center'],\
               height=DEFAULT['cylinder']['height'], radius=DEFAULT['cylinder']['radius'], width=DEFAULT['cylinder']['width'],\
               noise = DEFAULT['cylinder']['noise'], seed=DEFAULT['cylinder']['seed'], train = True):
        np.random.seed(seed=seed)
        seeds = np.random.randint(0, high=1000, size=2)
        if train:
            seed = seeds[0]
            # np.random.seed(seed=seed[0])
        else:
            seed = seeds[1]
            # np.random.seed(seed=seed[1])

        np.random.seed(seed=seed)
        z = np.random.uniform(low = -height/2, high = height/2, size = n_samples) + n_samples*[center[2]]
        r, theta = np.random.uniform(low = radius-width/2, high = radius+width/2, size = n_samples),\
                np.random.uniform(low = 0, high = 2*np.pi, size = n_samples)
        
        t = np.copy(theta)
        
        if noise != 0:
            for i in range(len(z)):
                z[i] += np.random.normal(0,noise)
            for i in range(len(r)):
                r[i] += np.random.normal(0,noise)
                
        x, y = r*np.sin(theta) + n_samples*[center[0]], r*np.cos(theta) + n_samples*[center[1]]
        x, y, z = x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)
        
        if horizontal == False:
            datacoordinates = (np.concatenate((x, y, z), axis = 1), t) # vertical cylinder
        else:
            datacoordinates = (np.concatenate((z, x, y), axis = 1), t) # horizontal cylinder

        return datacoordinates

# Building the cylinders datasets:

Cylinder1, color1 = Cylinder().sample(5000, height = 20, width = 1, radius = 10)
Cylinder2, color2 = Cylinder().sample(5000, height = 20, width = 1, radius = 5, noise = 0)
ConcentricCylinders, color_ConcentricCylinders = np.concatenate((Cylinder1, Cylinder2)) , np.concatenate((color1,color2))

Cylinder3, color3 = Cylinder().sample(5000, center = [0,15,0], height = 20, width = 1, radius = 5)
ParallelCylinders, color_ParallelCylinders = np.concatenate((Cylinder2, Cylinder3)) , np.concatenate((color2,color3))

Cylinder4, color4 = Cylinder().sample(5000, horizontal = True, height = 20, width = 1, radius = 5)
OrthogonalCylinders, color_OrthogonalCylinders = np.concatenate((Cylinder3, Cylinder4)) , np.concatenate((color3,color4))

Cylinder5, color5 = Cylinder().sample(5000, center = [0,10,0], height = 20, width = 1, radius = 5)
ParallelCylindersTouching, color_ParallelCylindersTouching = np.concatenate((Cylinder2, Cylinder5)) , np.concatenate((color2,color5))
OrthogonalCylindersTouching, color_OrthogonalCylindersTouching = np.concatenate((Cylinder5, Cylinder4)) , np.concatenate((color5,color4))

# Datasets to be used:

SimpleCylinder = (Cylinder2, color2)
ConcentricCylinders = (ConcentricCylinders, color_ConcentricCylinders)
ParallelCylinders = (ParallelCylinders, color_ParallelCylinders)
OrthogonalCylinders = (OrthogonalCylinders, color_OrthogonalCylinders)
ParallelCylindersTouching = (ParallelCylindersTouching, color_ParallelCylindersTouching)
OrthogonalCylindersTouching = (OrthogonalCylindersTouching, color_OrthogonalCylindersTouching)


class SwissRoll(DataSet):
    '''
    Swiss roll
    '''
    fancy_name = "Swiss Roll Dataset"

    __slots__ = []
    def __init__(self):
        pass

    def sample(self, n_samples, noise = DEFAULT['swissroll']['noise'], seed = DEFAULT['swissroll']['seed'], train = True):
        np.random.seed(seed=seed)
        seeds = np.random.randint(0, high=1000, size=2)
        if train:
            seed = seeds[0]
        else:
            seed = seeds[1]

        return datasets.make_swiss_roll(n_samples=n_samples, noise=noise, random_state=seed)
