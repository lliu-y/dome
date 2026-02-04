import random
from pathlib import Path
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from Register import Registers
from dataset.base import ImagePathDataset
from dataset.utils import get_image_paths_from_dir
from PIL import Image
import cv2
import os


@Registers.dataset.register_with_name('custom_single')
class CustomSingleDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        image_paths = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, stage))
        self.flip = dataset_config.flip if stage == 'train' else False
        self.to_normal = dataset_config.to_normal

        self.imgs = ImagePathDataset(image_paths, self.image_size, flip=self.flip, to_normal=self.to_normal)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        return self.imgs[i], self.imgs[i]


@Registers.dataset.register_with_name('custom_aligned')
class CustomAlignedDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        image_paths_ori = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, f'{stage}/B'))
        image_paths_cond = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, f'{stage}/A'))
        self.flip = dataset_config.flip if stage == 'train' else False
        self.to_normal = dataset_config.to_normal

        self.imgs_ori = ImagePathDataset(image_paths_ori, self.image_size, flip=self.flip, to_normal=self.to_normal)
        self.imgs_cond = ImagePathDataset(image_paths_cond, self.image_size, flip=self.flip, to_normal=self.to_normal)

    def __len__(self):
        return len(self.imgs_ori)

    def __getitem__(self, i):
        return self.imgs_ori[i], self.imgs_cond[i]


@Registers.dataset.register_with_name('custom_colorization_LAB')
class CustomColorizationLABDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        self.image_paths = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, stage))
        self.flip = dataset_config.flip if stage == 'train' else False
        self.to_normal = dataset_config.to_normal
        self._length = len(self.image_paths)

    def __len__(self):
        if self.flip:
            return self._length * 2
        return self._length

    def __getitem__(self, index):
        p = False
        if index >= self._length:
            index = index - self._length
            p = True

        img_path = self.image_paths[index]
        image = None
        try:
            image = cv2.imread(img_path)
            if self.to_lab:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        except BaseException as e:
            print(img_path)

        if p:
            image = cv2.flip(image, 1)
        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)
        image = torch.Tensor(image)
        image = image.permute(2, 0, 1).contiguous()

        if self.to_normal:
            image = (image - 127.5) / 127.5
            image.clamp_(-1., 1.)

        L = image[0:1, :, :]
        ab = image[1:, :, :]
        cond = torch.cat((L, L, L), dim=0)
        return image, cond


@Registers.dataset.register_with_name('custom_colorization_RGB')
class CustomColorizationRGBDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        self.image_paths = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, stage))
        self.flip = dataset_config.flip if stage == 'train' else False
        self.to_normal = dataset_config.to_normal
        self._length = len(self.image_paths)

    def __len__(self):
        if self.flip:
            return self._length * 2
        return self._length

    def __getitem__(self, index):
        p = False
        if index >= self._length:
            index = index - self._length
            p = True

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=p),
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])

        img_path = self.image_paths[index]
        image = None
        try:
            image = Image.open(img_path)
        except BaseException as e:
            print(img_path)

        if not image.mode == 'RGB':
            image = image.convert('RGB')

        cond_image = image.convert('L')
        cond_image = cond_image.convert('RGB')

        image = transform(image)
        cond_image = transform(cond_image)

        if self.to_normal:
            image = (image - 0.5) * 2.
            image.clamp_(-1., 1.)
            cond_image = (cond_image - 0.5) * 2.
            cond_image.clamp_(-1., 1.)

        image_name = Path(img_path).stem
        return (image, image_name), (cond_image, image_name)


@Registers.dataset.register_with_name('custom_inpainting')
class CustomInpaintingDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        self.image_paths = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, stage))
        self.flip = dataset_config.flip if stage == 'train' else False
        self.to_normal = dataset_config.to_normal
        self._length = len(self.image_paths)

    def __len__(self):
        if self.flip:
            return self._length * 2
        return self._length

    def __getitem__(self, index):
        p = 0.
        if index >= self._length:
            index = index - self._length
            p = 1.

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=p),
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])

        img_path = self.image_paths[index]
        image = None
        try:
            image = Image.open(img_path)
        except BaseException as e:
            print(img_path)

        if not image.mode == 'RGB':
            image = image.convert('RGB')

        image = transform(image)

        if self.to_normal:
            image = (image - 0.5) * 2.
            image.clamp_(-1., 1.)

        height, width = self.image_size
        mask_width = random.randint(128, 180)
        mask_height = random.randint(128, 180)
        mask_pos_x = random.randint(0, height - mask_height)
        mask_pos_y = random.randint(0, width - mask_width)
        mask = torch.ones_like(image)
        mask[:, mask_pos_x:mask_pos_x+mask_height, mask_pos_y:mask_pos_y+mask_width] = 0

        cond_image = image * mask

        image_name = Path(img_path).stem
        return (image, image_name), (cond_image, image_name)


@Registers.dataset.register_with_name('anime_colorization')
class AnimeColorizationDataset(Dataset):
    """
    动漫线稿上色数据集
    
    数据集目录结构:
        dataset/
        ├── train/
        │   ├── color/    # 彩色原图 (GT)
        │   └── sketch/   # 线稿图像
        ├── val/
        │   ├── color/
        │   └── sketch/
        └── test/
            ├── color/
            └── sketch/
    
    返回格式 (dict):
        - 'GT': [3, H, W] 目标彩图 (y in BBDM)
        - 'lineart': [1, H, W] 线稿 (条件, Concat到UNet)
        - 'distorted': [3, H, W] 参考图 (训练时TPS变形，验证/测试时=GT)
        - 'name': 文件名
    """
    
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.stage = stage
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        self.to_normal = dataset_config.to_normal
        
        self.flip = dataset_config.flip if stage == 'train' else False
        
        # 路径设置
        base_path = dataset_config.dataset_path
        self.color_path = os.path.join(base_path, stage, 'color')
        self.sketch_path = os.path.join(base_path, stage, 'sketch')
        
        # 加载文件列表并排序
        self.color_files = sorted(os.listdir(self.color_path))
        self.sketch_files = sorted(os.listdir(self.sketch_path))
        
        # 检查文件名匹配
        assert len(self.color_files) == len(self.sketch_files), \
            f"文件数量不匹配: color={len(self.color_files)}, sketch={len(self.sketch_files)}"
        
        self._length = len(self.color_files)
        
        # TPS变形: 仅训练时启用
        self.use_tps = getattr(dataset_config, 'use_tps', True) if stage == 'train' else False
        self.tps_scale = getattr(dataset_config, 'tps_scale', 0.2)
        
        # Transform: 分离RGB和灰度
        self.tf_color = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
        ])
        self.tf_sketch = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        if self.flip:
            return self._length * 2
        return self._length
    
    def __getitem__(self, index):
        # 处理flip索引
        do_flip = False
        if index >= self._length:
            index = index - self._length
            do_flip = True
        
        # 检查文件名匹配
        assert self.color_files[index][:-4] == self.sketch_files[index][:-4], \
            f"文件名不匹配: {self.color_files[index]} vs {self.sketch_files[index]}"
        
        filename = self.color_files[index]
        
        # 加载彩色图像 (GT)
        color_path = os.path.join(self.color_path, filename)
        color_img = Image.open(color_path).convert('RGB')
        
        # 加载线稿图像 (1通道灰度)
        sketch_path = os.path.join(self.sketch_path, filename)
        sketch_img = Image.open(sketch_path).convert('L')
        
        # 水平翻转
        if do_flip:
            color_img = color_img.transpose(Image.FLIP_LEFT_RIGHT)
            sketch_img = sketch_img.transpose(Image.FLIP_LEFT_RIGHT)
        
        # TPS变形生成参考图
        if self.use_tps:
            from dataset.tps_warp import warp_image
            distorted_img = warp_image(color_img, distortion_scale=self.tps_scale)
        else:
            distorted_img = color_img
        
        # 转换为Tensor
        color_tensor = self.tf_color(color_img)        # [3, H, W]
        sketch_tensor = self.tf_sketch(sketch_img)     # [1, H, W]
        distorted_tensor = self.tf_color(distorted_img) # [3, H, W]
        
        # 归一化到 [-1, 1]
        if self.to_normal:
            color_tensor = (color_tensor - 0.5) * 2.0
            color_tensor.clamp_(-1., 1.)
            sketch_tensor = (sketch_tensor - 0.5) * 2.0
            sketch_tensor.clamp_(-1., 1.)
            distorted_tensor = (distorted_tensor - 0.5) * 2.0
            distorted_tensor.clamp_(-1., 1.)
        
        # 返回dict格式
        return {
            'GT': color_tensor,           # [3, H, W] 目标彩图 (x0 in BBDM, 要重建的终点)
            'lineart': sketch_tensor,     # [1, H, W] 线稿 (repeat后作为y, 布朗桥起点)
            'distorted': distorted_tensor, # [3, H, W] 参考图 (A1: ref=GT, A3: TPS变形)
            'name': Path(filename).stem,  # 文件名 (无扩展名)
        }
