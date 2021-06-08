from matplotlib import pyplot as plt
import numpy as np

def plot_dataset_3d(dataset, view_x = 20, view_y = -40):
    matrix, colors = dataset
    fig = plt.figure(figsize = (8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(matrix[:, 0], matrix[:, 1], matrix[:, 2], c=colors, cmap=plt.cm.Spectral)
    ax.view_init(view_x, view_y)
    plt.show()

def plot_embedding_2d(instance, dataset):
    plt.figure(figsize = (8, 8))
    plt.scatter(instance.embedding_[:, 0], instance.embedding_[:, 1], s=5, c=dataset[1], cmap='Spectral')
    plt.title('embedding of the training set by {}'.format(type(instance).__name__), fontsize=12)
    plt.show()

def plot_dataset_2d(dataset, view_x = 20, view_y = -40):
    matrix, colors = dataset
    plt.figure(figsize = (8, 8))
    plt.scatter(matrix[:, 0], matrix[:, 1], s=5, c=colors, cmap='Spectral')
    plt.show()

def plot_losses(LOSSES, which = []):
    plt.figure(figsize = (8, 4))
    for x in which:
        plt.plot(LOSSES[x], label = x)
    plt.legend()
    plt.show()

def plot_1simplices(data, pairings, color, view_x = 20, view_y = -40, figuresize1 = 10, figuresize2 = 10, name = 'noname', path_root = None, knn = False, dpi = 200, show = True, angle = 5,cmap = plt.cm.Spectral):
    fig = plt.figure(figsize = (figuresize1, figuresize2))
    ax = plt.gca(projection="3d")
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=color, s=100, cmap=cmap)


    i = 0
    if pairings is None:
        pass
    else:
        for pairing in pairings:
            if knn:
                for ind in pairing:
                    ax.plot([data[i, 0], data[ind, 0]],
                            [data[i, 1], data[ind, 1]],
                            [data[i, 2], data[ind, 2]], color='grey')
            else:
                ax.plot([data[pairing[0], 0], data[pairing[1], 0]],
                        [data[pairing[0], 1], data[pairing[1], 1]],
                        [data[pairing[0], 2], data[pairing[1], 2]], color='grey')

            i += 1



    ax.view_init(angle, 90)
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])
    ax.margins(0, 0,0)

    #plt.axis('scaled')

    #find axis range

    axis_min = [min(data[:, i]) for i in [0,1,2]]
    axis_max = [max(data[:, i]) for i in [0, 1, 2]]
    margin = [(axis_max[i] - axis_min[i])*0.05 for i in [0, 1, 2]]

    axis_range = [np.array([axis_max[i]-margin[i], axis_max[i]+ margin[i]])for i in [0, 1, 2]]

    ax.set_xlim(np.array([axis_min[0]-margin[0], axis_max[0]+ margin[0]]))
    ax.set_ylim(np.array([axis_min[1]-margin[1], axis_max[1]+ margin[1]]))
    ax.set_zlim(np.array([axis_min[2]-margin[2], axis_max[2]+ margin[2]]))
    #ax.axis('equal')
    for line in ax.xaxis.get_ticklines():
        line.set_visible(False)
    for line in ax.yaxis.get_ticklines():
        line.set_visible(False)
    for line in ax.zaxis.get_ticklines():
        line.set_visible(False)

    ax.view_init(view_x, view_y)

    if path_root is not None:
        fig = ax.get_figure()
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace = 0, hspace = 0)
        fig.savefig(path_root+'btightplotsc_{}'.format(name)+'.pdf', dpi=dpi, bbox_inches='tight',
                    pad_inches=0)
        bbox = fig.bbox_inches.from_bounds(1, 1, 5, 5)
        fig.savefig(path_root + 'b5plotsc_{}'.format(name) + '.pdf', dpi=dpi,bbox_inches = bbox,
        pad_inches = 0)
        bbox = fig.bbox_inches.from_bounds(1, 1, 4, 4)
        fig.savefig(path_root + 'b4plotsc_{}'.format(name) + '.pdf', dpi=dpi,bbox_inches = bbox,
        pad_inches = 0)

        bbox = fig.bbox_inches.from_bounds(1, 1, 3, 3)
        fig.savefig(path_root + 'b3plotsc_{}'.format(name) + '.pdf', dpi=dpi,bbox_inches = bbox,
        pad_inches = 0)

        bbox = fig.bbox_inches.from_bounds(1, 1, 6, 6)
        fig.savefig(path_root + 'b6plotsc_{}'.format(name) + '.pdf', dpi=dpi,bbox_inches = bbox,
        pad_inches = 0)

    if show:
        plt.show()
    plt.close()