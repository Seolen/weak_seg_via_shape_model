import ipdb
import numpy as np

import torch
import math



class Gaussian2D():
    def __init__(self, mean=(0, 0), variance=(1, 1)):
        self.mean = mean
        self.variance = variance

    def forward(self, inputs1, inputs2):
        mu1, mu2 = self.mean
        sigma1, sigma2 = self.variance
        part1 = 1.0 / (2*math.pi*sigma1*sigma2)
        part2 = torch.pow((inputs1 - mu1), 2)/(sigma1**2) + torch.pow((inputs2 - mu2), 2)/(sigma2**2)
        results = part1 * torch.exp(-1.0/2 * part2)
        return results


def gauss_func():
    mean, variance = (0.5, 0), (0.5/3, 1.0/3)
    gauss = Gaussian2D(mean, variance)

    prob_np = np.array([
        [0.5, 0.5, 0.5],
        [0.1, 0.9, 0.7]]
    )
    sdf_np = np.array([
        [0, 0.5, -0.5],
        [0, 0.5, 0.5]]
    )
    prob, sdf = torch.from_numpy(prob_np), torch.from_numpy(sdf_np)
    uncertainty = gauss.forward(prob, sdf)
    print(uncertainty)
    # ipdb.set_trace()

    from scipy.stats import multivariate_normal
    mean, sigma3 = (0.5, 0), (0.5, 1.0)
    variance = (sigma3[0] / 3, sigma3[1] / 3)
    norm_func = multivariate_normal(mean, cov=[[1, 0], [0, 1]])
    v_uncertain_flat = np.array(
        [norm_func.pdf((ele_prob, ele_sdf)) for (ele_prob, ele_sdf) in zip(prob_np.flatten(), sdf_np.flatten())])
    v_uncertain = v_uncertain_flat.reshape(prob.shape)

    print(v_uncertain)
    ipdb.set_trace()

if __name__ == '__main__':
    gauss_func()

    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D
    #
    # d = np.random.randn(10000000, 2)
    # N = 30
    # x = np.linspace(0, 1, 100)
    # y = np.linspace(-1, 1, 100)
    # t = np.meshgrid(x, y)
    #
    # mean, variance = (0.5, 0), (0.5 / 3, 1.0 / 3)
    # gauss = Gaussian2D(mean, variance)
    # prob, sdf = torch.from_numpy(x), torch.from_numpy(y)
    # density = gauss.forward(prob, sdf)
    # density = density.numpy()
    #
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter(x, y, density, c='r', marker='o', depthshade=True)
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.set_zlabel("z")
    # plt.title("")
    # plt.show()