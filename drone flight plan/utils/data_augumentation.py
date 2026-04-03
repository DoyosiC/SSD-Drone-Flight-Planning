# GitHub：amdegroot/ssd.pytorch を参考にしています。
# MIT License
# Copyright (c) 2017 Max deGroot, Ellis Brown

import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random

def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]

def jaccard_numpy(box_a, box_b):
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1]))
    area_b = ((box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
    union = area_a + area_b - inter
    return inter / union

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels

class Lambda(object):
    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, boxes=None, labels=None):
        return self.lambd(img, boxes, labels)

class ConvertFromInts(object):
    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.float32), boxes, labels

class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, labels

class ToAbsoluteCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes = boxes.copy()
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height
        return image, boxes, labels

class ToPercentCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes = boxes.copy()
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height
        return image, boxes, labels

class Resize(object):
    def __init__(self, size=300):
        self.size = size

    def __call__(self, image, boxes=None, labels=None):
        image = cv2.resize(image, (self.size, self.size))
        return image, boxes, labels

class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower
        assert self.lower >= 0

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(0, 2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)
        return image, boxes, labels

class RandomHue(object):
    def __init__(self, delta=18.0):
        assert 0.0 <= delta <= 360.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(0, 2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, boxes, labels

class RandomLightingNoise(object):
    def __init__(self):
        self.perms = (
            (0, 1, 2), (0, 2, 1),
            (1, 0, 2), (1, 2, 0),
            (2, 0, 1), (2, 1, 0)
        )

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(0, 2):
            swap = self.perms[random.randint(0, len(self.perms))]
            shuffle = SwapChannels(swap)
            image = shuffle(image)
        return image, boxes, labels

class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, boxes=None, labels=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, boxes, labels

class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower
        assert self.lower >= 0

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(0, 2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, boxes, labels

class RandomBrightness(object):
    def __init__(self, delta=32):
        assert 0.0 <= delta <= 255.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(0, 2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, boxes, labels

class ToCV2Image(object):
    def __call__(self, tensor, boxes=None, labels=None):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), boxes, labels

class ToTensor(object):
    def __call__(self, cvimage, boxes=None, labels=None):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), boxes, labels

class RandomSampleCrop(object):
    def __init__(self):
        self.sample_options = (
            None,
            (0.1, None), (0.3, None), (0.7, None), (0.9, None),
            (None, None),
        )

    def __call__(self, image, boxes=None, labels=None):

        if boxes is None or len(boxes) == 0 or (hasattr(boxes, 'shape') and boxes.shape[0] == 0):
            return image, boxes, labels
        
        height, width, _ = image.shape
        while True:
            index = random.randint(0, len(self.sample_options))
            mode = self.sample_options[index]
            if mode is None:
                return image, boxes, labels
            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')
            for _ in range(50):
                current_image = image
                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)
                if h / w < 0.5 or h / w > 2:
                    continue
                left = random.uniform(0, width - w)
                top = random.uniform(0, height - h)
                rect = np.array([int(left), int(top), int(left + w), int(top + h)])
                overlap = jaccard_numpy(boxes, rect)
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2], :]
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])
                mask = m1 * m2
                if not mask.any():
                    continue
                current_boxes = boxes[mask, :].copy()
                current_labels = labels[mask]
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2], rect[:2])
                current_boxes[:, :2] -= rect[:2]
                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:], rect[2:])
                current_boxes[:, 2:] -= rect[:2]
                return current_image, current_boxes, current_labels

class Expand(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, boxes, labels):
        if random.randint(0, 2):
            return image, boxes, labels
        height, width, depth = image.shape
        ratio = random.uniform(1, 4)
        left = int(random.uniform(0, width * ratio - width))
        top = int(random.uniform(0, height * ratio - height))
        expand_image = np.zeros((int(height * ratio), int(width * ratio), depth), dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[top:top + height, left:left + width] = image
        image = expand_image
        boxes = boxes.copy()
        boxes[:, :2] += (left, top)
        boxes[:, 2:] += (left, top)
        return image, boxes, labels

class RandomMirror(object):
    def __call__(self, image, boxes, classes):
        _, width, _ = image.shape
        if random.randint(0, 2):
            image = image[:, ::-1]
            boxes = boxes.copy()
            x_min = boxes[:, 0].copy()
            x_max = boxes[:, 2].copy()
            boxes[:, 0] = width - x_max
            boxes[:, 2] = width - x_min
        return image, boxes, classes

class SwapChannels(object):
    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        return image[:, :, self.swaps]

class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, boxes, labels):
        im = image.copy()
        im, boxes, labels = self.rand_brightness(im, boxes, labels)
        if random.randint(0, 2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, boxes, labels = distort(im, boxes, labels)
        return self.rand_light_noise(im, boxes, labels)

class WithProb(object):
    """変換 t を確率 p で適用（画像のみ加工、boxes/labelsは無変更）"""
    def __init__(self, t, p=0.5):
        self.t = t
        self.p = p

    def __call__(self, image, boxes=None, labels=None):
        if random.random() < self.p:
            image = self.t(image)
        return image, boxes, labels

class AddGaussianNoise(object):
    """ガウシアンノイズを付与。std は 0-255 スケールで指定"""
    def __init__(self, mean=0.0, std=10.0):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        # image: np.uint8 (H,W,C) 前提
        noise = np.random.normal(self.mean, self.std, image.shape).astype(np.float32)
        noisy = image.astype(np.float32) + noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        return noisy

class AddSaltPepperNoise(object):
    """ソルト＆ペッパーノイズ（白/黒点）"""
    def __init__(self, amount=0.01, s_vs_p=0.5):
        self.amount = float(amount)
        self.s_vs_p = float(s_vs_p)

    def __call__(self, image):
        out = image.copy()
        H, W, C = out.shape
        num = int(self.amount * H * W)

        # salt
        num_salt = int(self.s_vs_p * num)
        coords_salt = (np.random.randint(0, H, num_salt),
                       np.random.randint(0, W, num_salt))
        out[coords_salt] = 255

        # pepper
        num_pepper = num - num_salt
        coords_pepper = (np.random.randint(0, H, num_pepper),
                         np.random.randint(0, W, num_pepper))
        out[coords_pepper] = 0

        return out

class JpegCompression(object):
    """JPEG再圧縮によるノイズ（ブロック歪み）"""
    def __init__(self, quality_min=30, quality_max=70):
        self.qmin = quality_min
        self.qmax = quality_max

    def __call__(self, image):
        q = random.randint(self.qmin, self.qmax)
        # OpenCVでJPEGエンコード→デコード
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), q]
        ok, enc = cv2.imencode('.jpg', image, encode_param)
        if not ok:
            return image
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        return dec if dec is not None else image
    
# ===== ここから追記（data_augumentation.pyの末尾あたりに） =====
# === data_augumentation.py に追記 ===
class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=10.0, prob=0.5):
        self.mean = mean
        self.std = std
        self.prob = prob

    def __call__(self, image, boxes=None, labels=None):
        if random.random() < self.prob:
            noise = np.random.normal(self.mean, self.std, image.shape).astype(np.float32)
            noisy_img = image.astype(np.float32) + noise
            noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
            return noisy_img, boxes, labels
        return image, boxes, labels


class AddSaltPepperNoise(object):
    """ソルト＆ペッパーノイズを追加（pの確率で実行）"""
    def __init__(self, prob=0.005, p=0.3):
        self.prob = prob  # 1ピクセルがノイズ化する確率
        self.p = p

    def __call__(self, image, boxes=None, labels=None):
        if random.rand() < self.p:
            h, w, c = image.shape
            num = int(h * w * self.prob)
            # salt
            ys = np.random.randint(0, h, num)
            xs = np.random.randint(0, w, num)
            image[ys, xs] = 255
            # pepper
            ys = np.random.randint(0, h, num)
            xs = np.random.randint(0, w, num)
            image[ys, xs] = 0
        return image, boxes, labels


class RandomGaussianBlur(object):
    """ガウシアンブラー（確率p、カーネルは3or5）"""
    def __init__(self, p=0.5, ksize_choices=(3, 5)):
        self.p = p
        self.ks = tuple(ksize_choices)

    def __call__(self, image, boxes=None, labels=None):
        if random.rand() < self.p:
            k = int(random.choice(self.ks))
            if k % 2 == 0:  # 念のため奇数化
                k += 1
            out = cv2.GaussianBlur(image, (k, k), 0)
            return out, boxes, labels
        return image, boxes, labels


class RandomJPEGCompression(object):
    """JPEG圧縮アーティファクト風（確率p, 品質q_min〜q_max）"""
    def __init__(self, p=0.5, q_min=30, q_max=70):
        self.p = p
        self.q_min = int(q_min)
        self.q_max = int(q_max)

    def __call__(self, image, boxes=None, labels=None):
        if random.rand() < self.p:
            q = int(random.uniform(self.q_min, self.q_max))
            # OpenCVはBGRのままでOK
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), q]
            ok, enc = cv2.imencode('.jpg', image, encode_param)
            if ok:
                dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
                if dec is not None:
                    return dec, boxes, labels
        return image, boxes, labels

class RandomGaussianNoise(object):
    def __init__(self, prob=0.5, sigma_range=(3,18)):
        self.prob = prob
        self.sigma_range = sigma_range
    def __call__(self, image, boxes=None, labels=None):
        if random.randint(0,2) and random.random() < self.prob:
            sigma = random.uniform(*self.sigma_range)
            noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
            image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(image.dtype)
        return image, boxes, labels


class RandomSaltPepperNoise(object):
    """ソルト＆ペッパーノイズ（確率p、ノイズ比率sp_ratio）"""
    def __init__(self, p=0.5, sp_ratio=0.005):
        self.p = p
        self.sp_ratio = float(sp_ratio)

    def __call__(self, image, boxes=None, labels=None):
        if random.rand() < self.p:
            out = image.copy()
            h, w = out.shape[:2]
            num = int(self.sp_ratio * h * w)

            # salt
            ys = np.random.randint(0, h, size=num)
            xs = np.random.randint(0, w, size=num)
            out[ys, xs] = 255

            # pepper
            ys = np.random.randint(0, h, size=num)
            xs = np.random.randint(0, w, size=num)
            out[ys, xs] = 0

            return out, boxes, labels
        return image, boxes, labels