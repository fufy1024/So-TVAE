import os
import pickle
import numpy as np
import torch

#initialize  c_means, c_sigma
def init_clusters(num_clusters=3, latent_size=64,
                  c_m_file='./cluster_means.pickle'):#[3,64]
    """Initialize clusters."""
    # initialize sigma as constant, mu drawn randomly
    c_sigma =0.2
    cluster_mu_matrix = []
    # generate or restore cluster matrix
    if os.path.exists(c_m_file):
        # load existing matrix
        with open(c_m_file, 'rb') as rf:
            c_means = pickle.load(rf)
    else:
        # generate clusters
        print("Generating clusters, saving to the {}".format(c_m_file))
        for id_cluster in range(num_clusters):
                cluster_item = 2*np.random.random_sample(
                    (1, latent_size)) - 1#[1, latent_size]
                cluster_item = cluster_item/(np.sqrt(
                    np.sum(cluster_item**2)))
                cluster_mu_matrix.append(cluster_item)    
        c_means = np.stack(cluster_mu_matrix)#[num_clusters, 1, latent_size]
        
        with open(c_m_file, 'wb') as wf:
            pickle.dump(c_means, wf)
        
    c_means= torch.Tensor(c_means).squeeze() 

    return c_means, c_sigma


