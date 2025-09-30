import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# --- Load results ---
df = pd.read_csv("results.csv")
X = df[['x0', 'x1']].values
labels = df['label'].values

# Define colors for clusters
colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:purple']

# Helper: plot covariance ellipse
def plot_cov_ellipse(mean, cov, ax, color='black'):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    width, height = 2 * np.sqrt(vals)   # 1-sigma ellipse
    ellip = Ellipse(xy=mean, width=width, height=height, angle=theta,
                    edgecolor=color, fc='none', lw=2)
    ax.add_patch(ellip)

# --- Create plot ---
fig, ax = plt.subplots(figsize=(8, 6))

for k in np.unique(labels):
    idx = labels == k
    ax.scatter(X[idx,0], X[idx,1], s=15, c=colors[k % len(colors)],
               label=f'Cluster {k}', alpha=0.7)

    # Replace these with the actual means/covs printed by your GMM run:
    if k == 0:
        mu = np.array([2.98063, 2.02346])
        Sigma = np.array([[0.257547, -0.109153], [-0.109153, 0.420752]])
    elif k == 1:
        mu = np.array([-3.21983, -0.253335])
        Sigma = np.array([[0.911263, 0.45928], [0.45928, 0.670229]])
    elif k == 2:
        mu = np.array([-2.48973, 0.859673])
        Sigma = np.array([[0.989131, 0.194806], [0.194806, 0.518723]])

    plot_cov_ellipse(mu, Sigma, ax, color=colors[k % len(colors)])

ax.set_title("GMM Clusters with Gaussian Ellipses")
ax.set_xlabel("x0")
ax.set_ylabel("x1")
ax.axis('equal')
ax.legend()
plt.show()

