'''
Author: GanJi
No.201518008629004

Spectral Clustering
'''
import os
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

location = os.path.split(os.path.realpath(__file__))[0] + '\\'
#location = 'D:\\'
data_name = 'scroll_data.mat'
#data_name = 'small_scroll_data.mat'
origin_fig = 'origin_3d_data.png'
le_fig = 'le.png'

calc_dist = lambda x, y: sum(i * i for i in (x - y))

def k_nearest_neigbor_weights(data, point, k_edge, varance = 10):
    dist_index = []
    dists = []
    m = len(data)
 
    for i in range(0, m):
        dists.append(calc_dist(data[i], point))
    dist_index =  np.argsort(dists)[1 : k_edge + 1]
    weights = [np.exp(- dists[i] * 1.0 / (2 * varance * varance)) for i in dist_index]
    
    return dist_index, weights

def calc_M(data, k_edge = 10, varance = 10):
    m = len(data)
    W = np.zeros((m, m), dtype = 'float')
    print '[calc_M] begin'
    for i in range(0, m):
        print '%4.2f%%'%(i * 100.0 / m)
        index, weights =  k_nearest_neigbor_weights(data, data[i], k_edge, varance)
        sum_weights = sum(weights)
        for j in range(0, len(index)):
            W[i, index[j]] = weights[j] / sum_weights

    M = np.mat(np.eye(m)) - np.mat(W)
    M = M.T * M
  
    print '[calc_M] end'

    return M

def get_LE(M, k):
    print '[LE] begin'
    eig_values, eig_vecs = np.linalg.eig(M)
    eig_values = eig_values.real
    eig_vecs = eig_vecs.real

    counts = 0
    T = []
    for i in np.argsort(eig_values):
        if eig_values[i] > 0.00000001:
            T.append( eig_vecs[:, i])
            counts = counts + 1
        if counts >= k:
            break

    T = np.array(np.hstack(T))
    print '[LE] end'
    return T

def plot_data_2d(data, title = '', save_fig = ''):
    plt.figure()
    T = np.arange(0, 2 * np.pi,  2 * np.pi / len(data))
    if title.strip() != '':
        plt.title(title)
    plt.scatter(data[:, 0], data[:, 1], s = 50, c = T)

    if save_fig.strip() != '':
        plt.savefig(save_fig)

    plt.show()

def plot_data_3d(data, title = '', save_fig = ''):
    fig = plt.figure()       
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)

    T = np.arange(0, 2 * np.pi,  2 * np.pi / len(data))
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], s = 50, c = T)

    if save_fig.strip() != '':
        plt.savefig(save_fig)
    plt.show()

def run():
    scroll_data_3d = sio.loadmat(location + data_name)['data']
    M = calc_M(scroll_data_3d, 20, 10)
    low_data = get_LE(M, 2)
    plot_data_3d(scroll_data_3d, 'scroll_data', location + origin_fig)
    plot_data_2d(low_data, 'LE', location + le_fig)

run()