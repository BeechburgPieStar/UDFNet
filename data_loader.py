from sklearn.utils import shuffle
import torch
import numpy as np
import scipy.io as scio
import args_parser
import os
from sklearn.model_selection import train_test_split

args = args_parser.args_parser()

def addZeroPadding(X, margin=2):
    newX = np.zeros((
        X.shape[0] + 2 * margin,
        X.shape[1] + 2 * margin,
        X.shape[2]), dtype=np.float32)  # Use float32
    newX[margin:X.shape[0]+margin, margin:X.shape[1]+margin, :] = X
    return newX

def createImgCubeGenerator(X, gt, pos_list, windowSize, batch_size2):
    """
    Generate image cubes in batches to manage memory usage effectively.
    """
    margin = (windowSize - 1) // 2
    zeroPaddingX = addZeroPadding(X, margin=margin)
    for i in range(0, len(pos_list), batch_size2):
        batch_pos = pos_list[i:i + batch_size2]
        data_patches = np.array([
            zeroPaddingX[i:i + windowSize, j:j + windowSize, :]
            for i, j in batch_pos], dtype=np.float32)  # Ensure float32 is used
        labels = np.array([gt[i, j] for i, j in batch_pos], dtype=np.float32)  # Ensure labels are float32 if necessary
        yield data_patches, labels

def createPos(shape:tuple, pos:tuple, num:int):
    """
    creatre pos list after the given pos
    """
    if (pos[0]+1)*(pos[1]+1)+num >shape[0]*shape[1]:
        num = shape[0]*shape[1]-( (pos[0])*shape[1] + pos[1] )
    return [(pos[0]+(pos[1]+i)//shape[1] , (pos[1]+i)%shape[1] ) for i in range(num) ]

def createPosWithoutZero(hsi, gt):
    """
    creatre pos list without zero labels
    """
    mask = gt > 0
    return [(i,j) for i , row  in enumerate(mask) for j , row_element in enumerate(row) if row_element]


def splitTrainTestSet(X, gt, test_samples_per_class, randomState):
    """
    Split data set based on a specific number of samples per class for the test set.
    :param X: The input features.
    :param gt: The ground truth labels, assumed to be augmented and three times the size of the original labels.
    :param test_samples_per_class: A dictionary with class labels as keys and the number of test samples desired for each class as values.
    :param randomState: The random state for reproducibility.
    :return: X_train, X_test, gt_train, gt_test
    """
    X_train, X_test, gt_train, gt_test = [], [], [], []
    for label, test_count in test_samples_per_class.items():
        # Find the indices of the current class
        indices = np.where(gt == label)[0]
        # Shuffle indices
        indices = shuffle(indices, random_state=randomState)
        # Split according to the specified number of test samples
        test_indices = indices[:test_count]
        train_indices = indices[test_count:]
        # Append to the train/test lists
        X_train.append(X[train_indices])
        X_test.append(X[test_indices])
        gt_train.append(gt[train_indices])
        gt_test.append(gt[test_indices])

    # Concatenate all class-specific splits
    X_train = np.concatenate(X_train)
    X_test = np.concatenate(X_test)
    gt_train = np.concatenate(gt_train)
    gt_test = np.concatenate(gt_test)

    # Shuffle the train and test sets to mix the classes
    X_train, gt_train = shuffle(X_train, gt_train, random_state=randomState)
    X_test, gt_test = shuffle(X_test, gt_test, random_state=randomState)

    return X_train, X_test, gt_train, gt_test



def createImgPatch(lidar, pos:list, windowSize=25):
    """
    return lidar Img patches
    """
    margin = (windowSize-1)//2
    zeroPaddingLidar = np.zeros((
      lidar.shape[0] + 2 * margin,
      lidar.shape[1] + 2 * margin
            ))
    zeroPaddingLidar[margin:lidar.shape[0]+margin, margin:lidar.shape[1]+margin] = lidar
    return np.array([zeroPaddingLidar[i:i+windowSize, j:j+windowSize] for i,j in pos ])

def minmax_normalize(array):
    amin = np.min(array)
    amax = np.max(array)
    return (array - amin) / (amax - amin)


def data_aug_single_batch(hsi_batch, sar_batch, labels_batch):
    Xh_aug, Xl_aug, y_aug = [], [], []
    noise_level = 0.02

    for hsi, sar, label in zip(hsi_batch, sar_batch, labels_batch):
        Xh_aug.append(hsi)
        Xl_aug.append(sar)
        y_aug.append(label)

        noise = np.random.normal(0.0, noise_level, size=hsi.shape)
        noise_sar = np.random.normal(0.0, noise_level, size=sar.shape)
        Xh_aug.append(np.flip(hsi + noise, axis=1))
        Xl_aug.append(np.flip(sar + noise_sar, axis=1))

        k = np.random.randint(4)
        Xh_aug.append(np.rot90(hsi, k=k))
        Xl_aug.append(np.rot90(sar, k=k))

        y_aug.extend([label, label])

    return np.array(Xh_aug, dtype=np.float32), np.array(Xl_aug, dtype=np.float32), np.array(y_aug, dtype=np.int8)


class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, hsi, sar, labels):
        self.len = labels.shape[0]
        self.hsi = hsi
        self.sar = sar
        self.labels = labels - 1
    def __getitem__(self, index):
        return self.hsi[index], self.sar[index], self.labels[index]
    def __len__(self):
        return self.len


def build_datasets(root, dataset, patch_size, batch_size):
    data_hsi = scio.loadmat(os.path.join(root, dataset, 'HSI.mat'))['HSI']
    data_sar = scio.loadmat(os.path.join(root, dataset, 'LiDAR.mat'))['LiDAR']
    data_traingt = scio.loadmat(os.path.join(root, dataset, 'gt.mat'))['gt']

    print("Total number of elements in data_traingt:", np.size(data_traingt))

    data_hsi = minmax_normalize(data_hsi)
    data_sar = minmax_normalize(data_sar)
    data_sar = data_sar.reshape((166, 600, 1))
    print(data_hsi.shape)
    print(data_sar.shape)
    print('--------------------------------------------------')
    print('createImgCube')

    pos_list = createPosWithoutZero(data_hsi, data_traingt)
    pos_list_length = len(pos_list)
    print("Number of positions in pos_list:", pos_list_length)


    hsi_all, sar_all, labels_all = [], [], []

    hsi_generator = createImgCubeGenerator(data_hsi, data_traingt, pos_list, patch_size, args.batch_size)
    sar_generator = createImgCubeGenerator(data_sar, data_traingt, pos_list, patch_size, args.batch_size)

    total_labels_count = 0
    aug_labels_count = 0
    for (hsi_batch, hsi_labels), (sar_batch, _) in zip(hsi_generator, sar_generator):
        current_batch_labels_count = len(hsi_labels)
        print("Current batch labels count:", current_batch_labels_count)
        total_labels_count += current_batch_labels_count

        hsi_aug, sar_aug, labels_aug = data_aug_single_batch(hsi_batch, sar_batch, hsi_labels)

        hsi_all.append(hsi_aug)
        sar_all.append(sar_aug)
        labels_all.append(labels_aug)

        current_aug_labels_count = len(labels_aug)
        print("Current batch labels count:", current_aug_labels_count)
        aug_labels_count += current_aug_labels_count

    print("Total labels count:", total_labels_count)
    print("aug labels count:", aug_labels_count)

    # 合并数组
    hsi_all = np.concatenate(hsi_all, axis=0)
    sar_all = np.concatenate(sar_all, axis=0)
    labels_all = np.concatenate(labels_all, axis=0)

    unique_labels = np.unique(labels_all)
    print('Unique labels in labels_all:', unique_labels)

    print('--------------------------------------------------')
    print('splitTrainTestSet')

    test_samples_per_class = {1: 3905*3, 2: 2778*3, 3: 374*3, 4: 8969*3, 5: 10317*3, 6: 3052*3}

    X_train, X_test, gt_train, gt_test = splitTrainTestSet(hsi_all, labels_all, test_samples_per_class, randomState=128)
    X_train_2, X_test_2, _, _ = splitTrainTestSet(sar_all, labels_all, test_samples_per_class, randomState=128)

    print('X_train:', X_train.shape)
    print('X_test:', X_test.shape)
    print('gt_train:', gt_train.shape)
    print('gt_test:', gt_test.shape)
    print('X_train_2:', X_train_2.shape)
    print('X_test_2:', X_test_2.shape)
    print('--------------------------------------------------')

    modified_path = os.path.join(root, dataset, 'modified')
    if not os.path.exists(modified_path):
        os.makedirs(modified_path)

    X_train = torch.tensor(X_train, dtype=torch.float)
    X_test = torch.tensor(X_test, dtype=torch.float)
    gt_train = torch.tensor(gt_train, dtype=torch.long)
    gt_test = torch.tensor(gt_test, dtype=torch.long)
    X_train_2 = torch.tensor(X_train_2, dtype=torch.float)
    X_test_2 = torch.tensor(X_test_2, dtype=torch.float)

    torch.save(X_train, os.path.join(modified_path, 'X_train_l.pt'))
    torch.save(X_test, os.path.join(modified_path, 'X_test_l.pt'))
    torch.save(gt_train, os.path.join(modified_path, 'gt_train_l.pt'))
    torch.save(gt_test, os.path.join(modified_path, 'gt_test_l.pt'))
    torch.save(X_train_2, os.path.join(modified_path, 'X_train_2_l.pt'))
    torch.save(X_test_2, os.path.join(modified_path, 'X_test_2_l.pt'))

    print("Creating dataloader")
    trainset = TensorDataset(X_train, X_train_2, gt_train)
    testset = TensorDataset(X_test, X_test_2, gt_test)
    train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader


def splitTrainTestSet(X, gt, testRatio, randomState=111):
    """
    random split data set
    """
    X_train, X_test, gt_train, gt_test = train_test_split(X, gt, test_size=testRatio, random_state=randomState, stratify=gt)
    return X_train, X_test, gt_train, gt_test

if __name__ == '__main__':
    args = args_parser.args_parser()
    train_loader, test_loader = build_datasets(args.root, args.dataset, args.patch_size, args.batch_size)