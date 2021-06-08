### This comes from https://github.com/jgalle29/simplicial/blob/master/algebraic.py and https://github.com/jgalle29/simplicial/blob/master/utils.py

import itertools

import numpy as np
import scipy as sp
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection

import networkx as nx
# import dionysus as dns

#import utils

def flatten_complex(K):
    flat_K = {}
    for kn in K:
        flat_K.update(kn)
    return flat_K

def create_kn(prev_k, neighs, nu, cut_alpha=0.):
    K = {}

    for vertices, mship in prev_k.items():
        vertices = list(vertices)
        nhoods = [neighs[_] for _ in vertices]
        candidates = set.intersection(*nhoods)

        for cand in candidates:
            sx_name = tuple(sorted(vertices + [cand]))

            if sx_name not in K:
                c_mship = min([mship] + [nu[min([x, cand]), max([x, cand])] for x in vertices])
                if c_mship >= cut_alpha:
                    K[sx_name] = c_mship

    return K

def create_simplicial_complex(nu, max_dim=None, verbose=False, cut_alpha=0.):

    graph = nu.tocoo()
    
    if max_dim is None:
        max_dim = np.inf
        
    n_pts = 1 + int(max([np.max(graph.row), np.max(graph.col)]))

    # K[i] is the set of i-simplices
    K = []

    # Initialize 0 and 1 simplices as given by nu
    #K.append({(_): 1 for _ in range(n_pts)})
    K.append({})
    
    # Create neighborhoods
    neighs = [set() for _ in range(n_pts)]
    K1 = {}
    for (x, y, val) in zip(graph.row, graph.col, graph.data):
        if val >= cut_alpha:
            neighs[x].add(y)
            neighs[y].add(x)
            K1[tuple(sorted([x, y]))] = val
    K.append(K1)

    # Build higher oder simplices hierarchically
    prev_k = K1

    dim = 2
    while prev_k and dim <= max_dim:
        prev_k = create_kn(prev_k, neighs, nu, cut_alpha)
        K.append(prev_k)
        if verbose:
            print([len(_) for _ in K])
        dim += 1

    # Remove empty complex
    if len(K[-1]) == 0:
        K.pop()

    return K

def plot_graph(X, graph, show_pts=True, fade_edges=False, edges=None):
    if edges is None:
        coo_graph = graph.tocoo()
        edges = list(zip(coo_graph.row, coo_graph.col))
        factor = 0.5 if fade_edges else 1
        colors = [(0.7, 0, 0.7, factor * _) for _ in coo_graph.data]
    else:
        colors = [(0.7, 0, 0.7, 1) for _ in range(len(edges))]
    
    ecol = [(X[i], X[j]) for (i,j) in edges]
    
    line_col = LineCollection(ecol, colors=colors)
    plt.gca().add_collection(line_col)
    
    alpha = 0.3 if show_pts else 0
    plt.plot(X[:, 0], X[:, 1], 'b.', markersize=1, alpha=alpha)


def mix_ppf(p, d0, d1, rv):

    q = np.zeros(p.shape)

    q[p < d0] = 0
    q[p > 1 - d1] = 1

    # Renormalize area in case of truncated distro
    psi = 1 / (rv.cdf(1) - rv.cdf(0))

    bdd_ix = np.where(np.logical_and(p >= d0, p <= 1 - d1))
    q[bdd_ix] = rv.ppf(((p[bdd_ix] / psi) - d0) / (1 - d0 - d1))

    return q

def mix_cdf(x, d0, d1, rv):

    p = np.zeros(x.shape)

    p[x == 0] = d0
    p[x == 1] = 1

    # Renormalize area in case of truncated distro
    psi = 1 / (rv.cdf(1) - rv.cdf(0))

    bdd_ix = np.where(np.logical_and(x > 0, x < 1))
    p[bdd_ix] = (1 - d0 - d1) * psi * rv.cdf(x[bdd_ix]) + d0

    return p


def find_cut_alphas(data, max_cuts=10, make_plots=False):
    kde = scipy.stats.gaussian_kde(data)
    x = np.linspace(0, 1, 100)
    cdf = [kde.integrate_box_1d(-0.1, _) for _ in x]
    
    ix_list = [np.argmin(np.abs(cdf - tgt)) for tgt in np.linspace(0, 1, max_cuts)] 
    cut_alphas = list(set(x[ix_list]))
    cdf_at_alphas = [kde.integrate_box_1d(-0.1, _) for _ in cut_alphas]
    
    if make_plots:

        plt.figure(figsize=(5, 4))
        _ = plt.hist(data, bins=50)
        plt.plot(x, np.sqrt(len(data)) * kde.evaluate(x), c='r')
        
        plt.figure(figsize=(5, 4))
        plt.plot(x, cdf, label="CDF")
        plt.plot(cdf, x, label="PPF")
        plt.plot(x, x)
        plt.plot(cut_alphas, cdf_at_alphas, '*')

        plt.legend()

    return sorted(cut_alphas)


# def find_cut_alphas(data, inner_eps=0.01, max_cuts=10, rv_type=scipy.stats.laplace, make_plots=False):

#     ix_less_1 = data < 1 - inner_eps
#     delta1 = 1 - (ix_less_1.sum() / len(data))
#     delta0 = 0 # (data < inner_eps).sum() / len(data)

#     #bdd_data = data[np.where(np.logical_and( data >= inner_eps, data <= 1 - inner_eps))]
#     bdd_data = data[ix_less_1]
#     fit_params = rv_type.fit(bdd_data)
#     rv = rv_type(fit_params[0], fit_params[1])

#     cut_alphas = mix_ppf(np.linspace(0, 1, max_cuts), delta0, delta1, rv)

#     # Remove repetitions but keep order
#     cut_alphas = np.array(sorted(set(cut_alphas)))
#     cut_alphas[cut_alphas < 0] = 0
#     cut_alphas[cut_alphas > 1] = 1

#     cdf_at_alphas = mix_cdf(cut_alphas, delta0, delta1, rv)

#     if make_plots:

#         plt.figure(figsize=(5, 4))
#         _ = plt.hist(bdd_data, bins=50)

#         x = np.linspace(inner_eps, 1 - inner_eps, 100)
#         plt.plot(np.linspace(inner_eps, 1 - inner_eps, 100), 5000 * rv.pdf(x), c='r')

#         x = np.linspace(0, 1, 100)
#         plt.figure(figsize=(5, 4))
#         plt.plot(x, mix_ppf(x, delta0, delta1, rv), label="PPF")
#         plt.plot(x, mix_cdf(x, delta0, delta1, rv), label="CDF")
#         plt.plot(x, x)
#         plt.plot(cut_alphas, cdf_at_alphas, '*')
#         plt.legend()

#     return cut_alphas, (delta0, delta1, fit_params)


def cut_complex(K, alphas):
    
    # Look at the 1-simplices onwards
    keys = list(itertools.chain.from_iterable([_.keys() for _ in K[1:]]))
    vals = np.array(list(itertools.chain.from_iterable([_.values() for _ in K[1:]])))

    cuts = []
    for alpha in alphas:
        cuts.append(np.where(vals > alpha - np.finfo(np.float32).eps)[0])

    return keys, vals, cuts

def comp_homology(f, verbose=False):
    m = dns.homology_persistence(f)
    dgms = dns.init_diagrams(m, f)
    res = {}
    for i, dgm in enumerate(dgms):
        pers_num = len([p for p in dgm if p.death == np.inf])
        res[i] = pers_num
        if dgm:
            if verbose:
                print("\tDimension:", i, str(pers_num))
    return res

def hom_snaps(keys, cuts, alphas):

    pts = set(itertools.chain.from_iterable([list(_) for _ in keys]))

    f = dns.Filtration()

    for pt in pts:
        f.append(dns.Simplex([pt]))

    hom_dict = {}
    for alpha, cut in zip(alphas[::-1], cuts[::-1]):
        for ix in cut:
            sx = keys[ix]
            f.append(dns.Simplex(sx))

        f.sort()
        #print('Alpha: ' + str(np.round(alpha, 3)) + ' ' + str(f))

        d = comp_homology(f)

        for _ in d:
            if _ in hom_dict:
                hom_dict[_].append((alpha, d[_]))
            else:
                hom_dict[_] = [(alpha, d[_])]

    # Clean keys which only contain zeros
    k2d = [_ for _ in hom_dict if max([foo[1] for foo in hom_dict[_]]) == 0]
    for _ in k2d:
        hom_dict.pop(_, None)

    return hom_dict

def disp_hom(hom_dict, scale_fn=lambda x:x):
    plt.figure(figsize=(5,4))
    for i in hom_dict:
        g = hom_dict[i]
        x = [_[0] for _ in g]
        y = [scale_fn(_[1]) for _ in g]
        plt.plot(x, y, '-*', label='H'+str(i))

    plt.legend()
    plt.grid(c='grey')
    plt.show()

    for i in hom_dict:
        print('Dimension ' + str(i) + ": " + str(hom_dict[i][-1][1]))

def make_weighted_graph(graph, fn=lambda x: 1/x):
    G = nx.Graph()
    g = graph.tocoo()
    ebunch = [(i, j, fn(g.data[ix])) for ix, (i, j) in enumerate(zip(g.row, g.col))]
    G.add_weighted_edges_from(ebunch)
    return G

def shortest_path(G, nu, source, end):
    try:
        pth = nx.dijkstra_path(G, source, end)
        ds, nus = path_costs(G, nu, pth)
        return pth, ds, nus
    except:
        return 3 * [None]

def path_costs(G, nu, pth):
    cts = [G.get_edge_data(pth[i], pth[i+1])['weight'] for i in range(len(pth)-1)]
    nus = [nu[pth[i], pth[i+1]] for i in range(len(pth)-1)]
    return cts, nus

def plot_2d_path(X, pth, times, embedding=None, y_train=None):

    if not embedding is None:
        utils.plot_embedding(embedding, y_train)
    else:
        plt.figure(figsize=(7, 7))
        
    xs, ys = zip(*X[pth])
    for i, _ in enumerate(pth):
        #plt.plot(xs[i:i+2], ys[i:i+2], '*--', alpha=t_alphas[i], color='blue')
        plt.plot(xs[i:i+2], ys[i:i+2], '*--', color=plt.cm.jet(times[i]))
        
        
    #plt.plot(xs, ys, '*--', lw=2, color='blue', alpha = alphas ms=10);
    plt.xlim(min(xs) - np.abs(min(xs)) * 0.001, max(xs) + np.abs(max(xs)) * 0.001)
    plt.ylim(min(ys) - np.abs(min(ys)) * 0.01, max(ys) + np.abs(max(ys)) * 0.01)
    plt.show()
    
def geodesic_interpolate(source, end, G, nu, X, embedding=None, y=None, cmap='gray_r'):
    
    pth, ds, nus = shortest_path(G, nu, source, end)
    
    geo_dist = None
    
    if pth != None:
        
        rows, cols = utils.make_grid_size(len(pth))
        
        # Append 0 cost to reach the initial state and normalize
        times = np.cumsum([0] + ds)
        times /= times[-1]

        if X.shape[1] == 2:
            #plt.plot(X[:,0], X[:,1], 'b.', markersize=5, alpha=0.3);
            plot_2d_path(X, pth, times)
        else:
            fig = plt.figure(figsize=(1.1*cols, 1.5*rows))  # width, height in inches
            #fig = plt.figure()  # width, height in inches
            for i, pt in enumerate(pth):
                sub = fig.add_subplot(rows, cols, i + 1)
                sub.imshow(X[pt, ...], cmap=cmap)
                sub.set_title('i = ' + str(pt) + '\nt = ' + str(np.round(times[i], 3)))
                sub.axis('off')
            plt.show()
        
        
        if (not embedding is None) and embedding.shape[1] == 2:
            plot_2d_path(embedding, pth, times, embedding, y)
            geo_dist = np.sum(np.sum(np.diff(embedding[pth, ...], axis=0)**2, axis=1)**(0.5))
        
        plt.figure(figsize=(5 , 2))
        plt.plot(times[1:],  nus, '-', markersize=3)
        plt.ylim(0, 1.05)
        plt.xlabel('Time')
        plt.ylabel('Strength')

        return pth, ds, nus, times, geo_dist

    else:
        print("No path exists between the given points!")
        return 5 * [None]

def sample_batch(X, K, mu, npts):
    batch_ix = np.random.choice(len(K), npts)
    res = [np.random.dirichlet(len(K[ix]) * [1], 1) @ X[K[ix]] for ix in batch_ix]
    return np.concatenate(res, axis=0), mu[batch_ix]

def noisy_connected_components(graph, pct=90):
    ccpts = sp.sparse.csgraph.connected_components(graph)
    bc = np.bincount(ccpts[1])
    if len(bc) == 1:
        print('Only found one connected component in graph!')
        return None, None, None, None
    else:
        res = {i:bc[i] for i in range(len(bc))}
        sorted_by_value = sorted(res.items(), key=lambda kv: kv[1], reverse=True)
        vals = [_[1] for _ in sorted_by_value]
        ncpts = len(np.where(vals > 1.0 * np.percentile(vals, pct))[0])
        good_ix = np.isin(ccpts[1], [sorted_by_value[i][0] for i in range(ncpts)])
        return ncpts, sorted_by_value[:ncpts], good_ix, ccpts[1]

def explore_graph_cut(graph, n_samples, threshold=0, pct=90, join_res=True):
    
    G = sp.sparse.csr_matrix(graph, copy=True)
    G.data *= (G.data >= threshold)
    G.eliminate_zeros()
    
    if G.nnz > 0:
        ncpts, good_vals, good_ix, ccpts = noisy_connected_components(G, pct)
        print('Found ' + str(ncpts) + ' components!')
        
        if ncpts is not None:
            props = [_[1] for _ in good_vals]
            props = np.array(props) / sum(props)
            
            res = []
            for ix, (k, _) in enumerate(good_vals):
                lsamples = int(1.5 * n_samples * props[ix])
                print(ix, k, lsamples)
                if lsamples > 0 and np.any(ccpts == k):
                    res.append(list(np.random.choice(np.squeeze(np.argwhere(ccpts == k)), lsamples)))

            if join_res:
                return list(itertools.chain.from_iterable(res))
            return res    

# from sklearn.neighbors import KDTree
# def clean_noise(Z, good_ix):
#     kdt = KDTree(Z[good_ix], leaf_size=30, metric='euclidean')
#     dist, ind = kdt.query(Z[~good_ix], k=1)
#     newZ = np.array(Z, copy=True)
#     newZ[~good_ix] = newZ[good_ix][np.squeeze(ind)]
#     return newZ        