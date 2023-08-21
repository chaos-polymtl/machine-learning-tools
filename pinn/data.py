# ============================================================================
# Script to gather data and set the index of experimental points
# Author : Valérie Bibeau, Polytechnique Montréal, 2023
# ============================================================================

# ---------------------------------------------------------------------------
# Imports
import torch
import numpy as np
import pandas as pd
# Set seed
torch.set_default_dtype(torch.float)
torch.manual_seed(1234)
np.random.seed(1234)
# ---------------------------------------------------------------------------

def read_data(file):
    """Read csv file

    Args:
        file (string): File name of database

    Returns:
        numpy array: Tensor of data points (lines are instances,
                                            first row is time,
                                            next rows are concentrations)
    """
    data = pd.read_csv(file, sep=',')
    data = data.replace(np.nan, 0.)
    
    C = data.to_numpy()
    C = C[:,1:]

    return C

def find_idx_C(t, C):
    """Find index of experimental points

    Args:
        t (numpy array): Collocation points
        C (numpy array): Tensor of data points

    Returns:
        list: Index of data points in array of collocation points
    """
    idx = []
    t_data = C[:,0]
    for ti in t_data:
        idx.append(np.where(t == ti)[0][0])
        
    return idx

def put_in_device(x, y, device):
    """Put tensors in CPU or CUDA device

    Args:
        x (numpy array): Tensor of time points
        y (numpy array): Tensor of concentrations points
        device (string): In which device training will be done (CPU or CUDA)

    Returns:
        tensor: Torch tensors
    """

    X = torch.from_numpy(x).float().to(device)
    Y = torch.from_numpy(y).float().to(device)

    return X, Y

def gather_data(files, mul):
    """Build X and Y tensors from data

    Args:
        files (string): Name of files containing data points
        mul (int): Multiplicator to set the number of collocation points
                   that fit the time serie

    Returns:
        tensor and list: Tensors of X and Y, list of indexation
    """
    
    C = read_data(files[0])
    t = np.linspace(0,int(C[-1,0]),int(C[-1,0]*mul+1)).reshape(-1,1)

    idx_C = find_idx_C(t, C)
    idx_y0 = [0]

    X = np.copy(t)
    Y = np.zeros((X.shape[0], C[:,1:].shape[1]))
    for i in range(Y.shape[1]):
        Y[idx_C,i] = C[:,i+1]

    len_t = len(t)
    for i in range(1,len(files)):
        new_C = read_data(files[i])
        new_t = np.linspace(0,int(new_C[-1,0]),int(new_C[-1,0]*mul+1)).reshape(-1,1)

        new_idx_C = find_idx_C(new_t, new_C)

        new_X = np.copy(new_t)
        X = np.concatenate((X, new_X), axis=0)
        new_Y = np.zeros((new_X.shape[0], new_C[:,1:].shape[1]))
        for k in range(new_Y.shape[1]):
            new_Y[new_idx_C,k] = new_C[:,k+1]
        Y = np.concatenate((Y, new_Y), axis=0)

        for j in range(len(new_idx_C)):
            new_idx_C[j] = new_idx_C[j] + len_t

        idx_C = idx_C + new_idx_C
        idx_y0 = idx_y0 + [len_t]
        len_t += len(new_t)
        
    return X, Y, idx_C, idx_y0