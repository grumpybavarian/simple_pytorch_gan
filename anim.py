import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from imageio import imread
import seaborn as sns

sns.set_style("whitegrid")
fig = plt.figure()
ax = plt.axes()
ax.set_xlim([-4, 4])
ax.set_ylim([0, 1])
fig.set_tight_layout(True)

print(fig.get_dpi())


# Plot a scatter that persists (isn't redrawn) and the initial line.
x = np.arange(0, 20, 0.1)
sns.distplot(np.random.normal(0, 1, 10000), hist=False, color='green', ax=ax)


def update(i):
    if len(ax.lines) > 1:
        ax.lines.pop(1)

    label = 'timestep {0}'.format(i)
    print(label)
    # Update the line and the axes (with a new xlabel). Return a tuple of
    # "artists" that have to be redrawn for this frame.
    dist = sns.distplot(data[i], ax=ax, hist=False, color='red')
    ax.set_title(label)
    return dist, ax

if __name__ == '__main__':
    data = np.load('dists.npy')
    print(data.shape)

    anim = FuncAnimation(fig, update, frames=np.arange(0, data.shape[0]), interval=10)

    # anim.save('line.gif', dpi=80, writer='imagemagick')

    plt.show()


