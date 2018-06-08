import os
import numpy as np
import matplotlib.pyplot as plt

import cutde.fullspace

import common

n = 50
gauss_params = (1.0, 0.0, 0.3)
sm = 1.0
nu = 0.25
folder = 'results/tris'
common.check_folder(folder)

def main():
    pts, rects = common.planar_mesh(n)
    m = common.rect_to_tri_mesh(pts, rects)

    n_tris = m[1].shape[0]
    tri_pts = m[0][m[1]]
    tri_centers = np.mean(tri_pts, axis = 1)

    plt.figure(figsize = (15,15))
    plt.triplot(m[0][:,0], m[0][:,2], m[1])
    plt.plot(tri_centers[:, 0], tri_centers[:, 2], '*')
    plt.savefig(os.path.join(folder, 'tri_setup.pdf'))
    # plt.show()

    dist = np.linalg.norm(tri_centers, axis = 1)
    strike_slip = common.gaussian(*gauss_params, dist)
    slip = np.zeros((strike_slip.shape[0], 3))
    slip[:,0] = strike_slip


    strain, stress = common.eval_tris(tri_centers, tri_pts, slip, sm, nu)

    common.plot_tris(m, stress[:,3], 'trisxy', folder)
    common.plot_tris(m, stress[:,5], 'trisyz', folder)

if __name__ == "__main__":
    main()
