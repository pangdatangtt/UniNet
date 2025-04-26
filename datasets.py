from torchvision import transforms
from PIL import Image
import os
import torch
import glob
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, ImageFolder
import numpy as np
# from noise import Simplex_CLASS
import cv2
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, ConcatDataset
from video_dataset import Video_DataLoader
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="importlib._bootstrap")


BTAD_PATH = os.path.abspath(os.path.join("D:\ws/btad"))

UnsupervisedAD = ['MVTec AD', 'BTAD', 'MVTec 3D-AD', "VisA", "APTOS", "ISIC2018", "OCT2017", 'ped2']
SupervisedAD = ["Kvasir-SEG", "CVC-ClinicDB", "CVC-ColonDB", "VAD"]

mvtec_list = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
              'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper', 'tile', 'wood']
# mvtec_list = ['transistor']

mvtec3d_list = ["bagel", "carrot", "cookie", "dowel", "foam",
                "peach", "potato", "tire", "rope", "cable_gland"]
# mvtec3d_list = ["carrot"]

btad_list = ["01", "02", "03"]
# btad_list = ['02']

visa_list = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1',
             'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']


def loading_dataset(c, dataset_name):
    train_dataloader, test_dataloader = None, None

    if dataset_name == 'MVTec 3D-AD' and c.setting == 'oc':
        train_dataloader = get_data_loader('train', c)
        test_dataloader = get_data_loader('test', c)
    elif dataset_name == 'BTAD' and c.setting == 'oc':
        train_data = BTADDataset(c, is_train=True)
        test_data = BTADDataset(c, is_train=False)
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=c.batch_size, shuffle=True,
                                                       pin_memory=True)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, pin_memory=True)
    elif dataset_name == 'MVTec AD' and c.setting == 'oc':
        # train_path = './mvtec/' + c._class_ + '/train'
        # data_transform, gt_transform = get_data_transforms(c.image_size, c.image_size)
        # train_data = NoiseMVTecDataset(root=train_path, transform=data_transform)
        # train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=c.batch_size, shuffle=True)
        train_data = MVTecDataset(c, is_train=True)
        test_data = MVTecDataset(c, is_train=False)
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=c.batch_size, shuffle=True,
                                                       pin_memory=True)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, pin_memory=True)

    elif dataset_name in ['MVTec AD', 'BTAD', 'MVTec 3D-AD', "VisA"] and c.setting == 'mc':
        data_transform, gt_transform = get_data_transforms(256, 256)
        train_data_list = []
        test_data_list = []

        lr = {}
        if dataset_name == "VisA":
            dataset_name = "visa"
            class_list = visa_list
            # lr = {"lr_s": 5e-3, "lr_t": 1e-6}
        elif dataset_name == 'MVTec AD':
            dataset_name = 'mvtec'
            class_list = mvtec_list
            lr = {"lr_s": 2e-3, "lr_t": 5e-5}
        elif dataset_name == 'BTAD':
            dataset_name = 'btad'
            class_list = btad_list
        else:
            dataset_name = 'mvtec_3d'
            class_list = mvtec3d_list

        for i, item in enumerate(class_list):
            train_path = '../data/{}/'.format(dataset_name) + item + '/train'
            test_path = '../data/{}/'.format(dataset_name) + item

            train_data = ImageFolder(root=train_path, transform=data_transform)
            train_data.classes = item
            train_data.class_to_idx = {item: i}
            train_data.samples = [(sample[0], i) for sample in train_data.samples]

            test_data = MultiMVTecDataset(root=test_path, transform=data_transform,
                                          gt_transform=gt_transform, phase="test")
            train_data_list.append(train_data)
            test_data_list.append(test_data)

        train_data = ConcatDataset(train_data_list)
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=c.batch_size, shuffle=True,
                                                       num_workers=1, drop_last=True)
        test_dataloader_list = [
            torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1)
            for test_data in test_data_list]

        return train_dataloader, test_dataloader_list, class_list, lr

    elif dataset_name in ["APTOS", "ISIC2018", "OCT2017"]:
        data_transform, gt_transform = get_data_transforms(c.image_size, 224) \
        if dataset_name == 'ISIC2018' else get_data_transforms(c.image_size, c.image_size)

        train_path = '../data/{}'.format(dataset_name)
        test_path = '../data/{}'.format(dataset_name)

        train_data = MedicalDataset(root=train_path, transform=data_transform, phase="train")
        test_data = MedicalDataset(root=test_path, transform=data_transform, phase="test")
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=c.batch_size, shuffle=True,
                                                       drop_last=False)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    elif dataset_name in ["ped2"]:
        h, w = c.image_size, c.image_size
        time_step = 0
        train_folder = 'video/ped2/training/frames'
        # Loading dataset
        train_dataset = Video_DataLoader(train_folder, transforms.Compose([transforms.ToTensor(), ]),
                                         resize_height=h, resize_width=w, time_step=time_step, c=3)

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=c.batch_size, shuffle=True,
                                                       drop_last=True, pin_memory=True)
        test_folder = 'video/ped2/testing/frames'

        # Loading dataset
        test_dataset = Video_DataLoader(test_folder, transforms.Compose([transforms.ToTensor(), ]),
                                        resize_height=h, resize_width=w, time_step=time_step, c=3)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                                      shuffle=False, drop_last=False)

    elif dataset_name in ["Kvasir-SEG", "CVC-ClinicDB", "CVC-ColonDB"]:
        from polyp_dataset import get_loader, test_dataset
        train_path = '../medical_segmentation/datasets/{}/train'.format(dataset_name)
        # train_path = '../medical_segmentation/datasets/Cross/train'

        image_root = '{}/images/'.format(train_path)
        gt_root = '{}/masks/'.format(train_path)
        train_dataloader = get_loader(image_root, gt_root, batchsize=c.batch_size,
                                      trainsize=c.image_size, augmentation=True)

        test_path = '../medical_segmentation/datasets/{}/test'.format(dataset_name)
        # test_path = '../medical_segmentation/datasets/Cross/test'

        image_root = '{}/images/'.format(test_path)
        gt_root = '{}/masks/'.format(test_path)
        test_dataloader = test_dataset(image_root, gt_root, c.image_size)

        num1 = len(os.listdir(gt_root))

        return train_dataloader, test_dataloader, num1

    elif dataset_name in ['vad']:
        train_data = VADDataset(c, is_train=True)
        test_data = VADDataset(c, is_train=False)
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=c.batch_size, shuffle=True,
                                                       pin_memory=True)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, pin_memory=True)

    return train_dataloader, test_dataloader


class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, c, is_train=True, dataset='mvtec'):
        self.dataset_path = '../data/' + dataset
        self.class_name = c._class_
        self.is_train = is_train
        # self.is_vis = c.is_vis
        self.input_size = (c.image_size, c.image_size)
        self.aug = False
        phase = 'train' if self.is_train else 'test'
        self.img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        self.gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')
        # load dataset
        self.x, self.y, self.mask, _ = self.load_dataset()
        # set transforms
        if is_train:
                self.transform_x = T.Compose([
                    T.Resize(self.input_size, InterpolationMode.LANCZOS),
                    T.ToTensor()])
        # test:
        else:
            self.transform_x = T.Compose([
                T.Resize(self.input_size, InterpolationMode.LANCZOS),
                T.ToTensor()])
        # mask
        self.transform_mask = T.Compose([
            T.Resize(self.input_size, InterpolationMode.NEAREST),
            T.ToTensor()])

        self.normalize = T.Compose([T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]
        # x = Image.open(x).convert('RGB')
        # if os.path.isfile(x):
        x = Image.open(x)

        if self.class_name in ['zipper', 'screw', 'grid']:  # handle greyscale classes
            x = np.expand_dims(np.array(x), axis=2)
            x = np.concatenate([x, x, x], axis=2)

            x = Image.fromarray(x.astype('uint8')).convert('RGB')
        #
        x = self.normalize(self.transform_x(x))
        #
        if y == 0:
            mask = torch.zeros([1, *self.input_size])
        else:
            mask = Image.open(mask)
            mask = self.transform_mask(mask)

        return x, y, mask

    def __len__(self):
        return len(self.x)

    def load_dataset(self):

        img_tot_paths = list()
        gt_tot_paths = list()
        tot_labels = list()
        tot_types = list()

        defect_types = os.listdir(self.img_dir)

        for defect_type in defect_types:
            # if self.is_vis and defect_type == "good":
                # continue
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_dir, defect_type) + "/*")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([None] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_dir, defect_type) + "/*")
                gt_paths = glob.glob(os.path.join(self.gt_dir, defect_type) + "/*")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))

        assert len(img_tot_paths) == len(tot_labels), "Something wrong with test and ground truth pair!"

        return img_tot_paths, tot_labels, gt_tot_paths, tot_types


class VADDataset(torch.utils.data.Dataset):
    def __init__(self, c, is_train=True):
        self.dataset_path = 'data/vad'
        self.is_train = is_train
        self.is_vis = False
        self.input_size = (c.image_size, c.image_size)
        self.aug = False
        phase = 'train' if self.is_train else 'test'
        self.img_dir = os.path.join(self.dataset_path, phase)
        # load dataset
        self.x, self.y = self.load_dataset()
        # set transforms
        if is_train:
            self.transform_x = T.Compose([
                T.Resize(self.input_size, InterpolationMode.LANCZOS),
                T.ToTensor()])
        # test:
        else:
            self.transform_x = T.Compose([
                T.Resize(self.input_size, InterpolationMode.LANCZOS),
                T.ToTensor()])

        self.normalize = T.Compose([T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]
        # x = Image.open(x).convert('RGB')
        # if os.path.isfile(x):
        x = Image.open(x)

        # x = np.expand_dims(np.array(x), axis=2)
        # x = np.concatenate([x, x, x], axis=2)

        # x = Image.fromarray(x.astype('uint8')).convert('RGB')
        x = self.normalize(self.transform_x(x))

        return x, y

    def __len__(self):
        return len(self.x)

    def load_dataset(self):

        img_tot_paths = list()
        tot_labels = list()
        unseen_labels = list()

        defect_types = os.listdir(self.img_dir)

        for defect_type in defect_types:
            if self.is_vis and defect_type == "good":
                continue
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_dir, defect_type) + "/*.png")
                img_tot_paths.extend(img_paths)
                tot_labels.extend([0] * len(img_paths))
            else:
                if defect_type != 'bad_unseen_defects':
                    img_paths = glob.glob(os.path.join(self.img_dir, defect_type) + "/*.png")
                    img_paths.sort()
                    img_tot_paths.extend(img_paths)
                    tot_labels.extend([1] * len(img_paths))
                else:
                    img_paths_ = glob.glob(os.path.join(self.img_dir, defect_type) + "/*.png")
                    img_paths_.sort()
                    img_tot_paths.extend(img_paths_)
                    tot_labels.extend([2] * len(img_paths_))

        assert len(img_tot_paths) == len(tot_labels), "Something wrong with test and ground truth pair!"

        return img_tot_paths, tot_labels,


class BTADDataset(torch.utils.data.Dataset):
    def __init__(self, c, is_train=True):
        assert c._class_ in btad_list, 'class_name: {}, should be in {}'. \
            format(c._class_, btad_list)
        self.dataset_path = BTAD_PATH
        self.class_name = c._class_
        self.is_train = is_train
        self.input_size = (c.image_size, c.image_size)
        # load dataset
        self.x, self.y, self.mask = self.load_dataset_folder()
        # set transforms
        if is_train:
            self.transform_x = T.Compose([
                T.Resize(self.input_size, InterpolationMode.LANCZOS),
                T.ToTensor()])
        # test:
        else:
            self.transform_x = T.Compose([
                T.Resize(self.input_size, InterpolationMode.LANCZOS),
                T.ToTensor()])
        # mask
        self.transform_mask = T.Compose([
            T.Resize(self.input_size, InterpolationMode.NEAREST),
            T.ToTensor()])

        self.normalize = T.Compose([T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]
        # x = Image.open(x).convert('RGB')
        x = Image.open(x)
        x = self.normalize(self.transform_x(x))
        #
        if y == 0:
            mask = torch.zeros([1, *self.input_size])
        else:
            mask = Image.open(mask)
            mask = self.transform_mask(mask)

        return x, y, mask

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y, mask = [], [], []

        img_dir = os.path.join(self.dataset_path, self.class_name, phase)  # data/Mvtec/bottle/train
        gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')  # data/Mvtec/bottle/gt

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:
            # load images
            img_type_dir = os.path.join(img_dir, img_type)  # data/Mvtec/bottle/train/good(ok)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir) if
                                     f.endswith(('.png', '.jpg', '.bmp'))])  # data/Mvtec/bottle/train/good/000.png...
            x.extend(img_fpath_list)

            # load gt labels
            if img_type == 'ok':
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)  # # data/Mvtec/bottle/train/broken_large...
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in
                                  img_fpath_list]  # ['000', '001', ...]
                pix = '.bmp' if self.class_name == '03' else '.png'
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + pix)
                                 for img_fname in img_fname_list]  # data/Mvtec/bottle/train/broken_large/000_maks.png
                mask.extend(gt_fpath_list)

        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y), list(mask)


class MultiMVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good' or defect_type == 'ok':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_path


# def get_data_transforms(size, isize):
#     data_transforms = transforms.Compose([Normalize(), \
#                                           ToTensor()])
#     gt_transforms = transforms.Compose([
#         transforms.Resize((size, size)),
#         transforms.ToTensor()])
#     return data_transforms, gt_transforms
def get_data_transforms(size, isize, mean_train=None, std_train=None):
    mean_train = [0.485, 0.456, 0.406] if mean_train is None else mean_train
    std_train = [0.229, 0.224, 0.225] if std_train is None else std_train
    data_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.CenterCrop(isize),
        transforms.Normalize(mean=mean_train,
                             std=std_train)])
    gt_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(isize),
        transforms.ToTensor()])
    return data_transforms, gt_transforms


class MedicalDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
        self.transform = transform
        self.phase = phase
        # load dataset
        self.img_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1
        self.num = 0

    def load_dataset(self):

        img_tot_paths = []
        tot_labels = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'NORMAL':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*")
                img_tot_paths.extend(img_paths)
                tot_labels.extend([0] * len(img_paths))
            else:
                if self.phase == 'train':
                    continue
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*")
                img_tot_paths.extend(img_paths)
                tot_labels.extend([1] * len(img_paths))

        return img_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, label = self.img_paths[idx], self.labels[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        return img, label, img_path


"--------------------------------------------------3D_dataset----------------------------------------------------------"

from utils import utils_3D

DATASETS_PATH = os.path.abspath(os.path.join("D:\ws\THF\data\Mvtec_3d"))


class MVTec_3D(torch.utils.data.Dataset):
    def __init__(self, split, class_name, img_size, centercrop):
        self.IMAGENET_MEAN = [0.485, 0.456, 0.406]
        self.IMAGENET_STD = [0.229, 0.224, 0.225]
        self.cls = class_name
        self.size = img_size
        self.centercrop = centercrop
        self.img_path = os.path.join(DATASETS_PATH, self.cls, split)
        self.rgb_transform = transforms.Compose(
            [
                transforms.CenterCrop((self.centercrop, self.centercrop)),
                transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)])


class MVTec_3D_Train(MVTec_3D):
    def __init__(self, class_name, img_size, centercrop):
        super().__init__(split="train", class_name=class_name, img_size=img_size, centercrop=centercrop)
        self.img_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):
        img_tot_paths = []
        tot_labels = []
        rgb_paths = glob.glob(os.path.join(self.img_path, 'good', 'rgb') + "/*.png")
        tiff_paths = glob.glob(os.path.join(self.img_path, 'good', 'xyz') + "/*.tiff")
        rgb_paths.sort()
        tiff_paths.sort()
        sample_paths = list(zip(rgb_paths, tiff_paths))
        img_tot_paths.extend(sample_paths)
        tot_labels.extend([0] * len(sample_paths))
        return img_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, label = self.img_paths[idx], self.labels[idx]
        rgb_path = img_path[0]
        tiff_path = img_path[1]
        img = Image.open(rgb_path).convert('RGB')

        img = self.rgb_transform(img)
        organized_pc = utils_3D.read_tiff_organized_pc(tiff_path)
        depth_map_3channel = np.repeat(utils_3D.organized_pc_to_depth_map(organized_pc)[:, :, np.newaxis], 3, axis=2)
        resized_depth_map_3channel = utils_3D.resize_organized_pc(depth_map_3channel, self.size)
        resized_organized_pc = utils_3D.resize_organized_pc(organized_pc, self.size)
        return (img, resized_organized_pc, resized_depth_map_3channel), label


class MVTec_3D_Test(MVTec_3D):
    def __init__(self, class_name, img_size, centercrop):
        super().__init__(split="test", class_name=class_name, img_size=img_size, centercrop=centercrop)
        self.gt_transform = transforms.Compose([
            transforms.CenterCrop((self.centercrop, self.centercrop)),
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()])
        self.img_paths, self.gt_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                rgb_paths = glob.glob(os.path.join(self.img_path, defect_type, 'rgb') + "/*.png")
                tiff_paths = glob.glob(os.path.join(self.img_path, defect_type, 'xyz') + "/*.tiff")
                rgb_paths.sort()
                tiff_paths.sort()
                sample_paths = list(zip(rgb_paths, tiff_paths))
                img_tot_paths.extend(sample_paths)
                gt_tot_paths.extend([0] * len(sample_paths))
                tot_labels.extend([0] * len(sample_paths))
            else:
                rgb_paths = glob.glob(os.path.join(self.img_path, defect_type, 'rgb') + "/*.png")
                tiff_paths = glob.glob(os.path.join(self.img_path, defect_type, 'xyz') + "/*.tiff")
                gt_paths = glob.glob(os.path.join(self.img_path, defect_type, 'gt') + "/*.png")
                rgb_paths.sort()
                tiff_paths.sort()
                gt_paths.sort()
                sample_paths = list(zip(rgb_paths, tiff_paths))

                img_tot_paths.extend(sample_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(sample_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label = self.img_paths[idx], self.gt_paths[idx], self.labels[idx]
        rgb_path = img_path[0]
        tiff_path = img_path[1]
        img_original = Image.open(rgb_path).convert('RGB')
        img = self.rgb_transform(img_original)

        organized_pc = utils_3D.read_tiff_organized_pc(tiff_path)
        depth_map_3channel = np.repeat(utils_3D.organized_pc_to_depth_map(organized_pc)[:, :, np.newaxis], 3, axis=2)
        resized_depth_map_3channel = utils_3D.resize_organized_pc(depth_map_3channel, self.size)
        resized_organized_pc = utils_3D.resize_organized_pc(organized_pc, self.size)

        if gt == 0:
            gt = torch.zeros(
                [1, resized_depth_map_3channel.size()[-2], resized_depth_map_3channel.size()[-2]])
        else:
            gt = Image.open(gt).convert('L')
            gt = self.gt_transform(gt)
            gt = torch.where(gt > 0.5, 1., .0)

        # if self.is_rd:
        #     return (img, resized_organized_pc, resized_depth_map_3channel), label, gt

        return (img, resized_organized_pc, resized_depth_map_3channel), label, gt[:1]


def get_data_loader(split, c):
    is_rd = c.is_rd if hasattr(c, "is_rd") else False
    if split in ['train']:
        dataset = MVTec_3D_Train(class_name=c._class_, img_size=c.image_size, centercrop=c.center_crop)
    elif split in ['test']:
        dataset = MVTec_3D_Test(class_name=c._class_, img_size=c.image_size, centercrop=c.center_crop,
                                is_vis=c.is_vis, is_rd=is_rd)

    is_train = split in ['train']
    batch_size = c.batch_size if is_train else 1
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=is_train,
                                              drop_last=is_train, pin_memory=True)
    return data_loader


def preprocess_for_3d(c):
    # crop off the background
    if c._class_ in ['potato']:
        c.center_crop = 384
    elif c._class_ in ['dowel', 'cable_gland']:
        c.center_crop = 256
    elif c._class_ in ['foam']:
        c.center_crop = 700
    else:
        c.center_crop = 900 if c._class_ == 'rope' else 512

    return c


class ToTensor(object):
    def __call__(self, image):
        try:
            image = torch.from_numpy(image.transpose(2, 0, 1))
        except:
            print('Invalid_transpose, please make sure images have shape (H, W, C) before transposing')
        if not isinstance(image, torch.FloatTensor):
            image = image.float()
        return image


class Normalize(object):
    """
    Only normalize images
    """

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, image):
        image = (image - self.mean) / self.std
        return image