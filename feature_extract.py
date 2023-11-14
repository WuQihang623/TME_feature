import os
import pickle
import numpy as np
import igraph as ig
import matplotlib.pyplot as plt
from collections import defaultdict
from GLCM import getGLCMFeatures
from leidenalg import find_partition, ModularityVertexPartition
from morphological import getMorphFeatures

def feature_extract(pkl_path, wsi_path, output_path):
    print(f"Processing {wsi_path}")
    wsi_name = os.path.splitext(os.path.basename(wsi_path))[0]
    output_path = os.path.join(output_path, wsi_name)

    with open(pkl_path, 'rb') as f:
        nucleusInfo = pickle.load(f)
        f.close()
    print()
    contours = nucleusInfo['contour'][:10000]
    types = nucleusInfo['type'][:10000]
    nucleusIds = [i for i in range(contours.shape[0])]
    morphfeatures = getMorphFeatures(nucleusIds, contours, process_n=1)
    glcmfeatures = getGLCMFeatures(wsi_path, nucleusIds, contours, pad=2, level=0, process_n=1)

    globalGraph = ig.Graph()
    globalGraph.add_vertices(len(nucleusIds), attributes={"nucleusId": nucleusIds, "Contour": contours, "CellType": types})
    for k, v in morphfeatures.items():
        if k != "nucleusId":
            globalGraph.vs[morphfeatures['nucleusId']]['Morph_' + k] = v

    for k, v in glcmfeatures.items():
        if k != "nucleusId":
            globalGraph.vs[glcmfeatures['nucleusId']]['Texture_' + k] = v

    # partition = find_partition(globalGraph, partition_type=ModularityVertexPartition)
    #
    # # 添加一个新的属性以存储聚类结果
    # globalGraph.vs['cluster'] = partition.membership
    # # 绘制图形
    # layout = globalGraph.layout('kk')  # 使用 Kamada-Kawai 布局
    # visual_style = {'vertex_color': globalGraph.vs['cluster'], 'vertex_size': 10, 'layout': layout}
    # fig, ax = plt.subplots()
    # ig.plot(globalGraph, **visual_style, target=ax)
    # plt.show()

    vertex_dataframe = globalGraph.get_vertex_dataframe()

    col_dist = defaultdict(list)
    cellType = ['T', 'I', 'N', 'S']
    for featname in vertex_dataframe.columns.values:
        for cell in cellType:
            col_dist[cell] += [featname] if featname != 'Contour' else []

    cellType_save = {'T': [1],
                     'I': [2],
                     'N': [4],
                     'S': [3]}

    os.makedirs(output_path, exist_ok=True)
    for i in col_dist.keys():
        vertex_csvfile = os.path.join(output_path, wsi_name + '_Feats_' + i + '.csv')
        save_index = vertex_dataframe['CellType'].isin(cellType_save[i]).values
        vertex_dataframe.iloc[save_index].to_csv(vertex_csvfile, index=False, columns=col_dist[i])

    print()



if __name__ == '__main__':
    pkl_path = "D:/SYMH/Results/463042.pkl"
    wsi_path = "D:/SYMH/WSI/463042.qptiff"
    output_path = "output"
    feature_extract(pkl_path, wsi_path, output_path)