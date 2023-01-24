import torch
import numpy as np

# simple function to put the data in the GPU memory
def cuda(x):
    return torch.FloatTensor(x).cuda()

def split(dataset, percentage_train, percentage_validation, percentage_test, permutation):
    assert percentage_train + percentage_validation + percentage_test == 1

    matrix, colors = dataset
    n = matrix.shape[0]
    if permutation == True:
        permutation = np.random.permutation(n)
    else:
        permutation = np.array(range(n))
    idx_train = int(percentage_train*n)
    idx_validation = int( (percentage_train+percentage_validation)*n )
    idx_test = int( (percentage_train+percentage_validation+percentage_test)*n )

    train = ( matrix[0:idx_train], colors[0:idx_train] )
    validation = ( matrix[idx_train:idx_validation], colors[idx_train:idx_validation] )
    test = ( matrix[idx_validation:], colors[idx_validation:] )

    return train, validation, test

def isomap_kernel(D): #input should be a distance matrix D
    N = D.shape[0] # number of points considered
    I = torch.eye(N).cuda()
    A = torch.ones(N,N).cuda()
    return -0.5*torch.matmul(torch.matmul(I-(1/N)*A,torch.matmul(D,D)),(I-(1/N)*A))
    