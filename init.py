# importing libraries
import glob

import scipy
import scipy.io as io
import scipy.spatial as sp
from matplotlib import cm as CM
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter

from image import *


# function to create density maps for images
def gaussian_filter_density(gt):
    print(gt.shape)
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    leafsize = 2048
    # build kdtree
    tree = sp.KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=4)

    print('generate density...')
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
        else:
            sigma = np.average(np.array(gt.shape)) / 2. / 2.  # case: 1 point
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    print('done.')
    return density


# setting the root to the Shanghai dataset you have downloaded
# change the root path as per your location of dataset
root = 'G:\\Projects\\Github\\CrowdCounting\\CSRNet-pytorch\\ShanghaiTech_Crowd_Counting_Dataset'
part_A_train = os.path.join(root, 'part_A\\train_data', 'images')
part_A_test = os.path.join(root, 'part_A\\test_data', 'images')
part_B_train = os.path.join(root, 'part_B\\train_data', 'images')
part_B_test = os.path.join(root, 'part_B\\test_data', 'images')
path_sets = [part_A_train, part_A_test]

img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)

index = 0
for img_path in img_paths:
    print(img_path)
    mat = io.loadmat(img_path.replace('.jpg', '.mat').replace('images', 'ground-truth').replace('IMG_', 'GT_IMG_'))
    img = plt.imread(img_path)
    k = np.zeros((img.shape[0], img.shape[1]))
    gt = mat["image_info"][0, 0][0, 0][0]
    for i in range(0, len(gt)):
        if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
            k[int(gt[i][1]), int(gt[i][0])] = 1
    index = index + 1
    print(f'{index}/{len(img_paths)}')
    k = gaussian_filter_density(k)
    with h5py.File(img_path.replace('.jpg', '.h5').replace('images', 'ground-truth'), 'w') as hf:
        hf['density'] = k

plt.imshow(Image.open(img_paths[0]))

gt_file = h5py.File(img_paths[0].replace('.jpg','.h5').replace('images','ground-truth'),'r')
groundtruth = np.asarray(gt_file['density'])
plt.imshow(groundtruth,cmap=CM.jet)

np.sum(groundtruth)

path_sets = [part_B_train, part_B_test]
img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)

# creating density map for part_b images
index = 0
for img_path in img_paths:
    print(img_path)
    mat = io.loadmat(img_path.replace('.jpg', '.mat').replace('images', 'ground-truth').replace('IMG_', 'GT_IMG_'))
    img = plt.imread(img_path)
    k = np.zeros((img.shape[0], img.shape[1]))
    gt = mat["image_info"][0, 0][0, 0][0]
    for i in range(0, len(gt)):
        if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
            k[int(gt[i][1]), int(gt[i][0])] = 1
    index = index + 1
    print(f'{index}/{len(img_paths)}')
    k = gaussian_filter_density(k)
    with h5py.File(img_path.replace('.jpg', '.h5').replace('images', 'ground-truth'), 'w') as hf:
        hf['density'] = k

from torchvision import transforms
transform=transforms.Compose([
                      transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
                  ])

