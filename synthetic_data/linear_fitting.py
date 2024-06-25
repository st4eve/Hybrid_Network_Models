from sklearn import cluster
from sklearn.svm import SVC, LinearSVC
from sklearn.decomposition import PCA
from data import generate_synthetic_dataset_easy, generate_synthetic_dataset_easy_raw
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import pandas as pd


df = pd.read_pickle('./dataframes/original_data.pkl')

train_data = df['train_data']
test_data = df['test_data']


palette = sns.color_palette('pastel')[2:]
sns.set_palette(palette)

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
xx, yy = np.meshgrid(np.arange(xmin, xmax, 0.01),
                     np.arange(ymin, ymax, 0.01))
Z = kmeans_pca.predict(np.c_[xx.ravel(), yy.ravel()])

df = pd.DataFrame(train_data_pca, columns=['x', 'y'])
df['label'] = train_data[1]
df['type'] = 'train'

df_test = pd.DataFrame(test_data_pca, columns=['x', 'y'])
df_test['label'] = test_data[1]
df_test['type'] = 'test'

df = pd.concat([df, df_test])

centroids = kmeans_pca.cluster_centers_

cmap = ListedColormap(sns.color_palette(palette[0:4], as_cmap=True))


sns.scatterplot(data=df, x='x', y='y', hue='label', style='type', palette=palette)
plt.imshow(Z.reshape(xx.shape),
              interpolation='nearest',
              extent=(xx.min(), xx.max(), yy.min(), yy.max()),
              cmap=cmap,
              aspect='auto', origin='lower',alpha=0.3)

plt.scatter(centroids[:,0], centroids[:,1], marker='x', s=100, c='white')

print('KMeans PCA')
print('Training Accuracy', np.sum((kmeans_pca.predict(train_data_pca) == train_data[1])) / len(train_data[1]))
print('Testing Accuracy', np.sum((kmeans_pca.predict(test_data_pca) == test_data[1]))/len(test_data[1]))


plt.savefig('./figures/pca_kmeans.pdf', bbox_inches='tight')
plt.close()

svm = LinearSVC(C=1, random_state=17)

svm.fit(train_data[0], train_data[1])

print('SVM')
print(f'Training Accuracy {svm.score(train_data[0], train_data[1]):.2f}')
print(f'Testing Accuracy {svm.score(test_data[0], test_data[1]):.2f}')


svm_pca = LinearSVC(C=1, random_state=17)

svm_pca.fit(train_data_pca, train_data[1])

print('SVM PCA')
print(f'Training Accuracy {svm_pca.score(train_data_pca, train_data[1]):.2f}')
print(f'Testing Accuracy {svm_pca.score(test_data_pca, test_data[1]):.2f}')


Z = svm_pca.predict(np.c_[xx.ravel(), yy.ravel()])

df = pd.DataFrame(train_data_pca, columns=['x', 'y'])
df['label'] = train_data[1]
df['type'] = 'train'

df_test = pd.DataFrame(test_data_pca, columns=['x', 'y'])
df_test['label'] = test_data[1]
df_test['type'] = 'test'

df = pd.concat([df, df_test])

sns.scatterplot(data=df, x='x', y='y', hue='label', style='type', palette=palette)

plt.imshow(Z.reshape(xx.shape),
                interpolation='nearest',
                extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                cmap=cmap,
                aspect='auto', origin='lower',
                alpha=0.3)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')


plt.savefig('./figures/pca_svm.pdf', bbox_inches='tight')
plt.show()