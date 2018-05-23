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

    label = 'timestep {0}'.format(10 * i)
    print(label)
    # Update the line and the axes (with a new xlabel). Return a tuple of
    # "artists" that have to be redrawn for this frame.
    dist = sns.distplot(data[10 * i], ax=ax, hist=False, color='red')
    ax.set_title(label)
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    return dist, ax

if __name__ == '__main__':
    data = np.load('dists.npy')

    anim = FuncAnimation(fig, update, frames=np.arange(0, int(data.shape[0]/10)), interval=150)

    anim.save('gan_demo.gif', dpi=80, writer='imagemagick')


