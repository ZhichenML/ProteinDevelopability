import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import umap.umap_ as umap
import hdbscan
import json
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

filename = './figures/pca.png'
figure_dir = os.path.dirname(filename)
if not os.path.exists(figure_dir):
    os.makedirs(figure_dir)

if __name__ == '__main__':
    # load data
    data = np.load('/public/home/gongzhichen/code/data/tap.npz', allow_pickle=True)
    train_X, valid_X, test_X, train_y, valid_y, test_y = data['train_X'], data['valid_X'], data['test_X'], data['train_Y'], data['valid_Y'], data['test_Y']

    x_std = StandardScaler().fit_transform(np.concatenate((train_X, valid_X, test_X), axis=0))
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(x_std)
    

    X_pca = np.vstack((x_pca.T, np.concatenate((train_y, valid_y, test_y), axis=0))).T
    X_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'label'])

    # visualize PCA
    plt.figure(figsize=(10, 10))
    plt.scatter(x_pca[:, 0], x_pca[:, 1], c=np.concatenate((train_y, valid_y, test_y), axis=0))
    plt.title(f'PCA {pca.explained_variance_ratio_}')
    plt.colorbar()
    plt.show()
    plt.savefig('./figures/PCA.png')
    plt.close()

    # visualize t-SNE
    x_tsne = TSNE(n_components=2, perplexity=50, learning_rate=200, n_iter=1000).fit_transform(x_std)
    X_tsne = np.vstack((x_tsne.T, np.concatenate((train_y, valid_y, test_y), axis=0))).T
    X_tsne = pd.DataFrame(X_tsne, columns=['TSNE1', 'TSNE2', 'label'])

    plt.figure(figsize=(10, 10))
    plt.scatter(X_tsne['TSNE1'], X_tsne['TSNE2'], c=X_tsne['label'])
    plt.title('t-SNE')
    plt.colorbar()
    plt.show()
    plt.savefig('./figures/t-SNE.png')
    plt.close()

    # visualize UMAP
    x_umap = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2).fit_transform(x_std)
    X_umap = np.vstack((x_umap.T, np.concatenate((train_y, valid_y, test_y), axis=0))).T
    X_umap = pd.DataFrame(X_umap, columns=['UMAP1', 'UMAP2', 'label'])

    plt.figure(figsize=(10, 10))
    plt.scatter(X_umap['UMAP1'], X_umap['UMAP2'], c=X_umap['label'])
    plt.title('UMAP')
    plt.colorbar()
    plt.show()
    plt.savefig('./figures/UMAP.png')
    plt.close()

    # visualize HDBSCAN
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=10, cluster_selection_method='eom', prediction_data=True).fit(x_std)
    labels = clusterer.labels_
    X_hdbscan = np.vstack((x_pca.T, labels)).T
    X_hdbscan = pd.DataFrame(X_hdbscan, columns=['x', 'y', 'label'])

    plt.figure(figsize=(10, 10))
    sns.scatterplot(x='x', y='y', hue='label', data=X_hdbscan, palette='Set1')
    plt.title('HDBSCAN')
    plt.show()
    plt.savefig('./figures/HDBSCAN.png')
    plt.close()

    # visualize K-means
    kmeans = KMeans(n_clusters=5, random_state=0).fit(x_std)
    labels = kmeans.labels_
    X_kmeans = np.vstack((x_pca.T, labels)).T
    X_kmeans = pd.DataFrame(X_kmeans, columns=['x', 'y', 'label'])

    plt.figure(figsize=(10, 10))
    sns.scatterplot(x='x', y='y', hue='label', data=X_kmeans, palette='Set1')
    plt.title('K-means')
    plt.show()
    plt.savefig('./figures/K-means.png')
    plt.close()

    # visualize silhouette score
    silhouette_scores = []
    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(x_std)
        labels = kmeans.labels_
        silhouette_scores.append(silhouette_score(x_std, labels))

    plt.plot(range(2, 10), silhouette_scores)
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette score')
    plt.title('Silhouette score for K-means')
    plt.show()
    plt.savefig('./figures/Silhouette_score_K-means.png')
    plt.close() 

