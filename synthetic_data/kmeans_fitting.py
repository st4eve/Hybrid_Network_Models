from sklearn import cluster
from sklearn.decomposition import PCA
from data import generate_synthetic_dataset_easy, generate_synthetic_dataset_easy_raw
from quantum_base_kerr import train_data, test_data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

train_data = (train_data[0], np.argmax(train_data[1], axis=1))
test_data = (test_data[0], np.argmax(test_data[1], axis=1))

kmeans = cluster.KMeans(n_clusters=4, init='k-means++', random_state=17, max_iter=10000, n_init=100).fit(train_data[0])


print('KMeans')
print(f'Training Accuracy {np.sum((kmeans.predict(train_data[0]) == train_data[1])) / len(train_data[1]):.2f}')
print(f'Testing Accuracy: {np.sum((kmeans.predict(test_data[0]) == test_data[1])) / len(test_data[1]):.2f}')

pca = PCA(n_components=2)
pca.fit(train_data[0])
train_data_pca = pca.transform(train_data[0])
test_data_pca = pca.transform(test_data[0])

kmeans_pca = cluster.KMeans(n_clusters=4, init='k-means++', random_state=17, max_iter=10000, n_init=100).fit(train_data_pca)

xmin, xmax = train_data_pca[:, 0].min() - 0.1, train_data_pca[:, 0].max() + 0.1
ymin, ymax = train_data_pca[:, 1].min() - 0.1, train_data_pca[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.arange(xmin, xmax, 0.1),
                     np.arange(ymin, ymax, 0.1))
Z = kmeans_pca.predict(np.c_[xx.ravel(), yy.ravel()])

df = pd.DataFrame(train_data_pca, columns=['x', 'y'])
df['label'] = train_data[1]
df['type'] = 'train'

df_test = pd.DataFrame(test_data_pca, columns=['x', 'y'])
df_test['label'] = test_data[1]
df_test['type'] = 'test'

df = pd.concat([df, df_test])

centroids = kmeans_pca.cluster_centers_

sns.scatterplot(data=df, x='x', y='y', hue='label', style='type', palette='tab10')
plt.imshow(Z.reshape(xx.shape),
              interpolation='nearest',
              extent=(xx.min(), xx.max(), yy.min(), yy.max()),
              cmap='tab10',
              aspect='auto', origin='lower')

plt.scatter(centroids[:,0], centroids[:,1], marker='x', s=100, c='white')

print('KMeans PCA')
print('Training Accuracy', np.sum((kmeans_pca.predict(train_data_pca) == train_data[1])) / len(train_data[1]))
print('Testing Accuracy', np.sum((kmeans_pca.predict(test_data_pca) == test_data[1]))/len(test_data[1]))

plt.savefig('./figures/pca.png')