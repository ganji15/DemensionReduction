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
origin_fig = 'origin_3d_data.png'
kpca_fig = 'KernelPCA.png'

def mean_zero(data):
    m = len(data)
    mean = sum(data) / m
    mean_zero_data = data - mean
    return mean_zero_data

def kernel(sample_1, sample_2, varance):
    induct = sum([i * i for i in (sample_1 - sample_2)])
    return np.exp(-induct / (2 * varance * varance))
 
def get_Kernel_cov_mat(mean_zero_data, varance):
    m = len(mean_zero_data)
    K = np.zeros((m, m), dtype = 'float')
    print '[Kernel] begin'
    for i in range(0, m):
        print '%4.2f%%'%(i * 100.0 / m)
        for j in range(0, m):
            K[i, j] = kernel(mean_zero_data[i], mean_zero_data[j], varance) 

    print '[Kernel] end'
    return np.mat(K)   
 
def get_principle_compoent(kernel_cov_mat, k):
    print '[Principle_compoent] begin'  
    eig_values, eig_vecs = np.linalg.eig(kernel_cov_mat)
    eig_values = eig_values.real
    eig_vecs = eig_vecs.real

    counts = 0
    T = []
    for i in np.argsort(eig_values)[::-1][:k]:
        T.append( eig_vecs[:, i] * np.sqrt( eig_values[i]))
        counts = counts + 1

    T = np.array(np.hstack(T))
    print '[Principle_compoent] end'

    return T
  
def KernelPCA(data, k, varance):
    m = len(data)
    mean_zero_data = mean_zero(data)
    K = get_Kernel_cov_mat(mean_zero_data, varance)
    low_data = get_principle_compoent(K, k)
    return low_data
    
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
    low_data = KernelPCA(scroll_data_3d, 2, 20)
    plot_data_3d(scroll_data_3d, 'scroll_data', location + origin_fig)
    plot_data_2d(low_data, 'KernelPCA', location + kpca_fig)

run()