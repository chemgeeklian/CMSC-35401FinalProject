import random
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import time

from sklearn.metrics import f1_score

def plot3D(X,Y):
    plt.rcParams.update({'font.size': 9})
    s=5
    fig = plt.figure(figsize=[3,3],dpi=300)
    ax = fig.add_subplot(111, projection='3d')

    s1 = ax.scatter(X[:,0], X[:,1], X[:,2], s=s, c = Y, cmap = plt.cm.coolwarm, alpha=.4)

    ax.set_xlabel('$z_0$')
    ax.set_ylabel('$z_1$')
    ax.set_zlabel('$z_2$')

    plt.yticks(rotation=30)
    plt.xticks(rotation=-15)

    ax.view_init(elev=10, azim=240)
    ax.dist = 11.4

    cbar = fig.colorbar(s1, ax = ax, pad=0.01, fraction=0.036,)
    cbar.ax.set_ylabel('r.e.')

    plt.tight_layout()
    plt.show()
    
    
def random_split(array, parts=2, seed = 0):
    n = len(array)
    random.seed(seed)
    idx = random.sample(range(n), n)
    chunk_size = n // parts
    chunks = [array[idx[i * chunk_size : (i + 1) * chunk_size]] for i in range(parts - 1)]
    chunks.append(array[idx[(parts - 1) * chunk_size:]])
    return chunks


def true(varible):
    return varible >.5 


def F1(y, y_pred):
    return f1_score(true(y), true(y_pred))