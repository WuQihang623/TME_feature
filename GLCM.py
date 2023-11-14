import sys

import cv2
import tqdm
import numpy as np
import multiprocessing as mp
from openslide import OpenSlide
import skimage.feature as skfeat
from collections import defaultdict

def getCellImg(slidePtr, bbox, pad=2, level=0):
    bbox = [[bbox[0], bbox[1]], [bbox[0]+bbox[2], bbox[1]+bbox[3]]]
    bbox = np.array(bbox)
    bbox[0] = bbox[0] - pad
    bbox[1] = bbox[1] + pad
    cellImg = slidePtr.read_region(location=bbox[0] * 2 ** level, level=level, size=bbox[1] - bbox[0])
    cellImg = np.array(cv2.cvtColor(np.asarray(cellImg), cv2.COLOR_RGB2GRAY))
    return cellImg

def getCellMask(contour, bbox, pad=2, level=0):
    if level != 0:
        raise KeyError('Not support level now')
    left, top, width, height = bbox
    # image = np.zeros((height + extention * 2, width + extention * 2), dtype=np.uint8)
    cellMask = np.zeros((height + pad * 2,
                         width + pad * 2),
                        dtype=np.uint8)
    contour = np.array(contour)
    contour[:, 0] = contour[:, 0] - left + pad
    contour[:, 1] = contour[:, 1] - top + pad
    cv2.drawContours(cellMask, [contour], 0, 1, -1)
    return cellMask

def mygreycoprops(P):
    # reference https://murphylab.web.cmu.edu/publications/boland/boland_node26.html
    (num_level, num_level2, num_dist, num_angle) = P.shape
    if num_level != num_level2:
        raise ValueError('num_level and num_level2 must be equal.')
    if num_dist <= 0:
        raise ValueError('num_dist must be positive.')
    if num_angle <= 0:
        raise ValueError('num_angle must be positive.')

    # normalize each GLCM
    P = P.astype(np.float64)
    glcm_sums = np.sum(P, axis=(0, 1), keepdims=True)
    glcm_sums[glcm_sums == 0] = 1
    P /= glcm_sums

    Pxplusy = np.zeros((num_level + num_level2 - 1, num_dist, num_angle))
    Ixplusy = np.expand_dims(np.arange(num_level + num_level2 - 1), axis=(1, 2))
    P_flip = np.flip(P, axis=0)
    for i, offset in enumerate(range(num_level - 1, -num_level2, -1)):
        Pxplusy[i] = np.trace(P_flip, offset)
    SumAverage = np.sum(Ixplusy * Pxplusy, axis=0)
    Entropy = - np.sum(Pxplusy * np.log(Pxplusy + 1e-15), axis=0)
    SumVariance = np.sum((Ixplusy - Entropy) ** 2 * Pxplusy, axis=0)

    Ix = np.tile(np.arange(num_level).reshape(-1, 1, 1, 1), [1, num_level2, 1, 1])
    Average = np.sum(Ix * P, axis=(0, 1))
    Variance = np.sum((Ix - Average) ** 2 * P, axis=(0, 1))
    return SumAverage, Entropy, SumVariance, Average, Variance

def SingleGLCMFeatures(args):
    nucleusIds, contours, wsiPath, pad, level = args
    slidePtr = OpenSlide(wsiPath)
    # Use wsipath as parameter because multiprocess can't use pointer like the object OpenSlide() as parameter
    featuresDict = defaultdict(list)
    featuresDict['nucleusId'] = nucleusIds
    contours = tqdm.tqdm(contours, file=sys.stdout)
    for contour in contours:
        bbox = cv2.boundingRect(contour)
        cellImg = getCellImg(slidePtr, bbox, pad, level)
        cellmask = getCellMask(contour, bbox, pad).astype(np.bool_)
        cellImg[~cellmask] = 0

        outMatrix = skfeat.graycomatrix(cellImg, [1], [0])
        outMatrix[0, :, ...] = 0
        outMatrix[:, 0, ...] = 0

        dissimilarity = skfeat.graycoprops(outMatrix, 'dissimilarity')[0][0]
        homogeneity = skfeat.graycoprops(outMatrix, 'homogeneity')[0][0]
        # energy = skfeat.greycoprops(outMatrix, 'energy')[0][0]
        ASM = skfeat.graycoprops(outMatrix, 'ASM')[0][0]
        contrast = skfeat.graycoprops(outMatrix, 'contrast')[0][0]
        correlation = skfeat.graycoprops(outMatrix, 'correlation')[0][0]
        SumAverage, Entropy, SumVariance, Average, Variance = mygreycoprops(outMatrix)

        featuresDict['ASM'] += [ASM]
        featuresDict['Contrast'] += [contrast]
        featuresDict['Correlation'] += [correlation]
        # featuresDict['Dissimilarity'] += [dissimilarity]
        featuresDict['Entropy'] += [Entropy[0][0]]
        featuresDict['Homogeneity'] += [homogeneity]
        # featuresDict['Energy'] += [energy] #Delete because similar with ASM
        # featuresDict['Average'] += [Average[0][0]]
        # featuresDict['Variance'] += [Variance[0][0]]
        # featuresDict['SumAverage'] += [SumAverage[0][0]]
        # featuresDict['SumVariance'] += [SumVariance[0][0]]

        featuresDict['IntensityMean'] += [cellImg[cellmask].mean()]
        featuresDict['IntensityStd'] += [cellImg[cellmask].std()]
        featuresDict['IntensityMax'] += [cellImg[cellmask].max().astype('int16')]
        featuresDict['IntensityMin'] += [cellImg[cellmask].min().astype('int16')]
        # featuresDict['IntensitySkewness'] += [stats.skew(cellImg.flatten())] # Plan to delete this feature
        # featuresDict['IntensityKurtosis'] += [stats.kurtosis(cellImg.flatten())] # Plan to delete this feature
    return featuresDict

def getGLCMFeatures(wsi_path, nucleusIds, contours, pad, level, process_n):
    assert process_n > 0
    if process_n == 1:
        featuresDict = SingleGLCMFeatures([nucleusIds, contours, wsi_path, pad, level])
    else:
        featuresDict = defaultdict(list)
        batch_size = len(nucleusIds) // process_n
        args = [[nucleusIds[i * batch_size: (i + 1) * batch_size], contours[i * batch_size: (i + 1) * batch_size], wsi_path, pad, level] for i in range(process_n)]
        with mp.Pool(process_n) as p:
            ans = p.map(SingleGLCMFeatures, args)
        for q_info in ans:
            for k, v in zip(q_info.keys(), q_info.values()):
                featuresDict[k] += v
    return featuresDict
