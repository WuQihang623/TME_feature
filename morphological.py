# 提取形态特征
'''
Area, AreaBbox, CellEccentricities, Circularity, Elongation, Extent,
MajorAxisLength, MinorAxisLength, Perimeter, Solidity, CurveMean, CurvMin, CurvStd
'''
import sys

import cv2
import tqdm
import numpy as np
import multiprocessing as mp
from openslide import OpenSlide
import skimage.feature as skfeat
from collections import defaultdict
from skimage.measure import regionprops

# 计算曲率
def getCurvature(contour, n_size=5):
    contour = np.array(contour)
    contour_circle = np.concatenate([contour, contour[0:1]], axis=0)
    dxy = np.diff(contour_circle, axis=0)

    samplekeep = np.zeros((len(contour)), dtype=np.bool_)
    samplekeep[0] = True
    flag = 0
    for i in range(1, len(contour)):
        if np.abs(contour[i] - contour[flag]).sum() > 2:
            flag = i
            samplekeep[flag] = True

    contour = contour[samplekeep]
    contour_circle = np.concatenate([contour, contour[0:1]], axis=0)
    dxy = np.diff(contour_circle, axis=0)

    ds = np.sqrt(np.sum(dxy ** 2, axis=1, keepdims=True))
    ddxy = dxy / ds
    ds = (ds + np.roll(ds, shift=1)) / 2
    Cxy = np.diff(np.concatenate([ddxy, ddxy[0:1]], axis=0), axis=0) / ds
    Cxy = (Cxy + np.roll(Cxy, shift=1, axis=0)) / 2
    k = (ddxy[:, 1] * Cxy[:, 0] - ddxy[:, 0] * Cxy[:, 1]) / ((ddxy ** 2).sum(axis=1) ** (3 / 2))

    curvMean = k.mean()
    curvMin = k.min()
    curvMax = k.max()
    curvStd = k.std()

    n_protrusion = 0
    n_indentation = 0
    if n_size > len(k):
        n_size = len(k) // 2
    k_circle = np.concatenate([k[-n_size:], k, k[:n_size]], axis=0)
    for i in range(n_size, len(k_circle) - n_size):
        neighbor = k_circle[i - 5:i + 5]
        if k_circle[i] > 0:
            if k_circle[i] == neighbor.max():
                n_protrusion += 1
        elif k_circle[i] < 0:
            if k_circle[i] == neighbor.min():
                n_indentation += 1
    n_protrusion /= len(contour)
    n_indentation /= len(contour)

    return curvMean, curvStd, curvMax, curvMin, n_protrusion, n_indentation

def getRegionPropFromContour(contour, bbox, extention=2):
    # 利用regionprops来获取细胞核的面积、周长、中心坐标等信息
    left, top, width, height = bbox
    # image = np.zeros((height + extention * 2, width + extention * 2), dtype=np.uint8)
    image = np.zeros((height + extention * 2,
                      width + extention * 2),
                     dtype=np.uint8)
    contour = np.array(contour)
    contour[:, 0] = contour[:, 0] - left + extention
    contour[:, 1] = contour[:, 1] - top + extention
    cv2.drawContours(image, [contour], 0, 1, -1)
    # TODO: check contour coords
    regionProp = regionprops(image)[0]
    return regionProp

def getSingleMorphFeatures(args):
    nucleusIds, contours = args
    featuresDict = defaultdict(list)
    featuresDict['nucleusId'] = nucleusIds
    contours = tqdm.tqdm(contours, file=sys.stdout)
    for contour in contours:
        bbox = cv2.boundingRect(contour)
        regionProps = getRegionPropFromContour(contour, bbox)
        featuresDict['Area'] += [regionProps.area]
        featuresDict['AreaBbox'] += [regionProps.area_bbox]
        featuresDict['CellEccentricities'] += [regionProps.eccentricity] # 离心率
        featuresDict['Circularity'] += [(4 * np.pi * regionProps.area) / (regionProps.perimeter ** 2)] # 圆形度
        featuresDict['Elongation'] += [regionProps.major_axis_length / regionProps.minor_axis_length] # 伸长度
        featuresDict['Extent'] += [regionProps.extent] # 占外接矩形的比例
        # featuresDict['FeretDiameterMax'] += [regionProps.feret_diameter_max]
        featuresDict['MajorAxisLength'] += [regionProps.major_axis_length] # 主轴长度
        featuresDict['MinorAxisLength'] += [regionProps.minor_axis_length] # 次轴长度
        # featuresDict['Orientation'] += [regionProps.orientation]
        featuresDict['Perimeter'] += [regionProps.perimeter] # 周长
        featuresDict['Solidity'] += [regionProps.solidity] # 实心度，面积与凸包面积的比率

        curvMean, curvStd, curvMax, curvMin, n_protrusion, n_indentation = getCurvature(contour)
        featuresDict['CurvMean'] += [curvMean]
        featuresDict['CurvStd'] += [curvStd]
        featuresDict['CurvMax'] += [curvMax]
        featuresDict['CurvMin'] += [curvMin]
    return featuresDict

# 执行列表的切分
def getMorphFeatures(nucleusIds, contours, process_n=1):
    assert process_n > 0
    if process_n == 1:
        featuresDict = getSingleMorphFeatures([nucleusIds, contours])
    else:
        featuresDict = defaultdict(list)
        batch_size = len(nucleusIds) // process_n
        args = [[nucleusIds[i*batch_size: (i+1)*batch_size], contours[i*batch_size: (i+1)*batch_size]] for i in range(process_n)]
        with mp.Pool(process_n) as p:
            ans = p.map(getSingleMorphFeatures, args)
        for q_info in ans:
            for k, v in zip(q_info.keys(), q_info.values()):
                featuresDict[k] += v
    return featuresDict

