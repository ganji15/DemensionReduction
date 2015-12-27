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
data_name = 'scroll_data.mat'
#data_name = 'small_scroll_data.mat'
origin_fig = 'scroll.png'
eigmap_fig = 'eigmap.png'
llp_fig = 'llp.png'

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

def calc_Lsym(data, k_edge = 10, varance = 10):
    m = len(data)
    W = np.zeros((m, m), dtype = 'float')
    print '[calc_L] begin'
    for i in range(0, m):
        print '%4.2f%%'%(i * 100.0 / m)
        index, weights =  k_nearest_neigbor_weights(data, data[i], k_edge, varance)
        for j in range(0, len(index)):
            W[i, index[j]] = weights[j]

    D = np.zeros((m, m), dtype = 'float')
    for i in range(0, m):
        D[i, i] = sum(W[i])

    D = np.mat(D)
    W = np.mat(W)
    L = D - W
    D_inv_sqrt = np.zeros((m, m), dtype = 'float')
    for i in range(0, m):
        D_inv_sqrt[i, i] = 1.0 / np.sqrt(D[i, i])
    D_inv_sqrt = np.mat(D_inv_sqrt)
    Lsym = D_inv_sqrt * L * D_inv_sqrt
    print '[calc_L] end'

    return Lsym

def get_LP_eigmap(Lsym, k):
    print '[LP_eig] begin'
    eig_values, eig_vecs = np.linalg.eig(Lsym)
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
    print '[LP_eig] end'
    return T

def get_LLP_eigmap(Lsym, data, k):
    print '[LLP_eig] begin'
    Lsym_X = np.mat(data).I * Lsym * np.mat(data)
    eig_values, eig_vecs = np.linalg.eig(Lsym_X)
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

    T = np.hstack(T)
    print '[LLP_eig] end'
    return T

def LLP_low_dim_data(data, LLP_eigmap):
    low_dim_data = np.mat(data) * LLP_eigmap  
    return np.array(low_dim_data)

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

def LaplacianEigenmap(data, dim, Lsym):
    t_data = get_LP_eigmap( Lsym, dim)
    return t_data

def LocallityPreservingProjection(data, dim, Lsym):
    T = get_LLP_eigmap( Lsym, data, dim)
    t_data = LLP_low_dim_data(data, T)
    return t_data

def run():
    scroll_data_3d = sio.loadmat(location + data_name)['data']
    Lsym = calc_Lsym(scroll_data_3d, 20, 10)
    eigmap_data = LaplacianEigenmap(scroll_data_3d, 2, Lsym)
    llp_data = LocallityPreservingProjection(scroll_data_3d, 2, Lsym)
    plot_data_3d(scroll_data_3d, 'scroll_data', location + origin_fig)
    plot_data_2d(eigmap_data, 'laplace eigmap', location + eigmap_fig)
    plot_data_2d(llp_data, 'llp', location + llp_fig)
       
run()