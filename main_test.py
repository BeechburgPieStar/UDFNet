import time
import os
import torch.optim
import args_parser
from time import *
import scipy.io as scio
import numpy as np
import scipy.io as sio
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
import torch

args = args_parser.args_parser()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print (args)

def addZeroPadding(X, margin=2):
    """
    add zero padding to the image
    """
    newX = np.zeros((
      X.shape[0] + 2 * margin,
      X.shape[1] + 2 * margin,
      X.shape[2]
            ))
    newX[margin:X.shape[0]+margin, margin:X.shape[1]+margin, :] = X
    return newX

def minmax_normalize(array):
    amin = np.min(array)
    amax = np.max(array)
    return (array - amin) / (amax - amin)

def mainqmf(model):
    end = time()
    if args.dataset == 'Berlin':
        args.hsi_bands = 244
        args.sar_bands = 4
        args.num_class = 8
    elif args.dataset == 'Augsburg':
        args.hsi_bands = 180
        args.sar_bands = 4
        args.num_class = 7
    elif args.dataset == 'MUUFL':
        args.hsi_bands = 64
        args.sar_bands = 2
        args.num_class = 11
    elif args.dataset == 'trento':
        args.hsi_bands = 63
        args.sar_bands = 1
        args.num_class = 6

    data_hsi = scio.loadmat(os.path.join(args.root, args.dataset, 'HSI.mat'))['HSI']
    data_sar = scio.loadmat(os.path.join(args.root, args.dataset, 'LiDAR.mat'))['LiDAR']
    data_gt = scio.loadmat(os.path.join(args.root, args.dataset, 'gt.mat'))['gt']

    height, width, c = data_hsi.shape
    data_hsi = minmax_normalize(data_hsi)
    data_sar = minmax_normalize(data_sar)
    data_sar = data_sar.reshape((166, 600, 1))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    margin = (args.patch_size - 1) // 2
    data_hsi = addZeroPadding(data_hsi, margin)
    data_sar = addZeroPadding(data_sar, margin)
    data_gt = np.pad(data_gt, ((margin, margin), (margin, margin)), 'constant', constant_values=(0, 0))

    idx, idy = np.where(data_gt != 0)
    labelss = np.array([0])

    batch = 200
    num = 10
    total_batch = int(len(idx) / batch + 1)
    print('Total batch number is :', total_batch)

    for j in range(int((len(idx) - (len(idx) % batch)) / batch + 1)):
        if int(100 * j // total_batch) == num:
            print('... ... ', int(num), '% batch handling ... ...')
            num = num + 10
        if batch * (j + 1) > len(idx):
            num_cat = len(idx) - batch * j
        else:
            num_cat = batch

        tmphsi = np.array([data_hsi[idx[j * batch + i] - margin:idx[j * batch + i] +
                                                                margin + 1,
                           idy[j * batch + i] - margin:idy[j * batch + i] + margin + 1, :] for i in range(num_cat)])
        tmpsar = np.array([data_sar[idx[j * batch + i] - margin:idx[j * batch + i] +
                                                                margin + 1,
                           idy[j * batch + i] - margin:idy[j * batch + i] + margin + 1, :] for i in range(num_cat)])
        if tmphsi.size == 0:
            print(f"Batch {j} is empty. Skipping...")
            continue
        tmphsi = torch.FloatTensor(tmphsi.transpose(0, 3, 1, 2)).to(device)
        tmpsar = torch.FloatTensor(tmpsar.transpose(0, 3, 1, 2)).to(device)

        prediction, _, _, _, _ = model(tmphsi, tmpsar)
        labelss = np.hstack([labelss, np.argmax(prediction.detach().cpu().numpy(), axis=1)])
    print('... ... ', int(100), '% batch handling ... ...')
    labelss = np.delete(labelss, [0])
    new_map = np.zeros((height, width))
    for i in range(len(idx)):
        new_map[idx[i] - margin, idy[i] - margin] = labelss[i] + 1
    scio.savemat(args.result_path + '/result.mat', {'output': new_map})
    print('Finish!!!')
    end2 = time()
    minutes = int((end2 - end) / 60)
    seconds = int((end2 - end) - minutes * 60)
    print("Running time：", minutes, "m", seconds, "s")

    predicted = sio.loadmat(os.path.join(args.result_path, 'result.mat'))['output']
    actual = scio.loadmat(os.path.join(args.root, args.dataset, 'gt.mat'))['gt']

    assert predicted.shape == actual.shape, "Predicted and actual data shapes do not match."

    predicted_flat = predicted.flatten()
    actual_flat = actual.flatten()

    mask = actual_flat > 0
    predicted_flat = predicted_flat[mask]
    actual_flat = actual_flat[mask]

    cm = confusion_matrix(actual_flat, predicted_flat)

    oa = accuracy_score(actual_flat, predicted_flat)

    user_accuracy = np.diag(cm) / np.sum(cm, axis=1)
    aa = np.mean(user_accuracy)
    kappa = cohen_kappa_score(actual_flat, predicted_flat)

    results = f"总体精度 (OA): {oa * 100:.2f}%\n"
    results += f"平均精度 (AA): {aa * 100:.2f}%\n"
    results += "每个类别的精度:\n"
    for j, acc in enumerate(user_accuracy, start=1):
        results += f"类别 {j}: {acc * 100:.2f}%\n"
    results += f"Kappa系数: {kappa:.4f}\n"

    result_file_path = os.path.join(args.result_path, 'result.txt')

    os.makedirs(args.result_path, exist_ok=True)
    with open(result_file_path, "w") as f:
        f.write(results)
    print('OA:', oa * 100)
    print('AA:', aa * 100)
    print('kappa:', kappa)
