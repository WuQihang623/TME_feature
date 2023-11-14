# T-SNE降维
```python
import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
path = 'singlecell_output/463042/463042_Feats_I.csv'
pd = pandas.read_csv(path)
tsne = TSNE(n_components=2, random_state=42) # n_components: 降维维度， random_state:随机种子
data = np.array(pd.drop("nucleusId", axis=1))
X_tsne = tsne.fit_transform(data)

kmeans = KMeans(n_clusters=3, random_state=42)
y_kmeans = kmeans.fit_predict(X_tsne)

# 绘制 t-SNE 降维后的数据，并根据 KMeans 聚类结果着色
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_kmeans, cmap='viridis', alpha=0.7)
plt.colorbar()
plt.show()
```