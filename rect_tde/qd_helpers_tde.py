import cppimport.import_hook
import qd_newton

def plot_fields(m, field, levels = None, cmap = 'seismic', symmetric_scale = False, ds = None, figsize = None):
    field_reshape = field.reshape(m.tris.shape[0],3,-1)
    n_fields = field_reshape.shape[2]
    if figsize is None:
        figsize = (6 * n_fields,5)
    plt.figure(figsize = figsize)
    if ds is None:
        ds = range(n_fields)
    for d in ds:
        plt.subplot(1, n_fields, d + 1)
        pt_field = np.empty((m.pts.shape[0], n_fields))
        pt_field[m.tris] = field_reshape
        f = pt_field[:,d]
        f_levels = levels
        if f_levels is None:
            min_f = np.min(f)
            max_f = np.max(f)
            scale_f = np.max(np.abs(f))
            if scale_f == 0.0:
                scale_f = 1.0
            min_f -= 1e-13 * scale_f
            max_f += 1e-13 * scale_f
            if symmetric_scale:
                min_f = -max_f
            f_levels = np.linspace(min_f, max_f, 21)
        cntf = plt.tricontourf(m.pts[:,0], m.pts[:,2], m.tris, f, cmap = cmap, levels = f_levels, extend = 'both')
        #plt.tricontour(m.pts[:,0], m.pts[:,2], m.tris, f, levels = levels, extend = 'both', linestyles = 'solid', linewidths = 0.75, colors = '#333333')
        plt.colorbar(cntf)
        #plt.axis('equal')
    plt.show()