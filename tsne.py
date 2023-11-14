import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

path = 'results/feature_clu_dead_r2_retraining_0924-SYSSecond.xlsx'
pd_data = pd.read_excel(path)
pd_data = pd_data.drop("文件名", axis=1)

print()