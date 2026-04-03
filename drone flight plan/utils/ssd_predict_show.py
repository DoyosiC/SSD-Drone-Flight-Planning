# utils/ssd_predict_show.py
"""
第2章SSDで予測結果を画像として描画するクラス
"""

import numpy as np
import matplotlib.pyplot as plt 
import cv2
import torch

from utils.ssd_model import DataTransform

class SSDPredictShow():
    """SSDでの予測と画像の表示をまとめて行うクラス"""

    def __init__(self, eval_categories, net):
        self.eval_categories = eval_categories  # クラス名
        self.net = net  # SSDネットワーク

        color_mean = (104, 117, 123)  # (BGR)の色の平均値
        input_size = 300  # 画像のinputサイズを300×300にする
        self.transform = DataTransform(input_size, color_mean)

    def show(self, image_file_path, data_confidence_level):
        """
        物体検出の予測結果を表示をする関数。
        """
        rgb_img, predict_bbox, pre_dict_label_index, scores = self.ssd_predict(
            image_file_path, data_confidence_level)

        self.vis_bbox(rgb_img, bbox=predict_bbox, label_index=pre_dict_label_index,
                      scores=scores, label_names=self.eval_categories)

    def ssd_predict(self, image_file_path, data_confidence_level=0.5):
        """
        SSDで予測させる関数。（画像ファイル入力）
        """
        img = cv2.imread(image_file_path)  # [高さ][幅][色BGR]
        height, width, channels = img.shape  # 画像のサイズを取得
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 画像の前処理（アノテなし）
        phase = "val"
        img_transformed, boxes, labels = self.transform(img, phase, "", "")
        img = torch.from_numpy(img_transformed[:, :, (2, 1, 0)].copy()).permute(2, 0, 1)

        # 推論
        self.net.phase = "inference"
        self.net.eval()
        x = img.unsqueeze(0).to(next(self.net.parameters()).device)
        # with torch.no_grad():
        #     detections = self.net(x)

        with torch.no_grad():
            if x.is_cuda:
                from torch.cuda.amp import autocast
                with autocast():
                    detections = self.net(x)
            else:
                detections = self.net(x)


        if isinstance(detections, tuple):
            detections = detections[0]

        # confidenceで抽出
        predict_bbox, pre_dict_label_index, scores = [], [], []
        detections = detections.cpu().detach().numpy()
        if detections.shape[0] == 0 or detections.shape[1] == 0:
            return rgb_img, [], [], []

        find_index = np.where(detections[:, 0:, :, 0] >= data_confidence_level)
        detections_found = detections[find_index]
        for i in range(len(find_index[1])):
            cl = int(find_index[1][i])
            if cl > 0:  # 背景以外
                sc = float(detections_found[i][0])
                bbox = detections_found[i][1:] * np.array([width, height, width, height], dtype=np.float32)
                pre_dict_label_index.append(cl - 1)
                predict_bbox.append(bbox)
                scores.append(sc)
        return rgb_img, predict_bbox, pre_dict_label_index, scores

    def vis_bbox(self, rgb_img, bbox, label_index, scores, label_names):
        """
        物体検出の予測結果を画像で表示させる関数。（matplotlib）
        """
        if len(bbox) == 0 or len(label_index) == 0:
            print("No detection found.")
            return

        num_classes = len(label_names)
        colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()
        plt.figure(figsize=(10, 10))
        plt.imshow(rgb_img)
        currentAxis = plt.gca()

        for i, bb in enumerate(bbox):
            label_name = label_names[label_index[i]]
            color = colors[label_index[i] % num_classes]
            if scores is not None and i < len(scores):
                sc = scores[i]
                display_txt = '%s: %.2f' % (label_name, sc)
            else:
                display_txt = '%s: ans' % (label_name)
            xy = (float(bb[0]), float(bb[1]))
            width = float(bb[2] - bb[0])
            height = float(bb[3] - bb[1])
            currentAxis.add_patch(plt.Rectangle(
                xy, width, height, fill=False, edgecolor=color, linewidth=2))
            currentAxis.text(xy[0], xy[1], display_txt, bbox={'facecolor': color, 'alpha': 0.5})
        plt.show()

    # ======== ここから新規：フレーム入力＆OpenCV描画 ========

    def ssd_predict_from_frame(self, frame_bgr, data_confidence_level=0.5):
        """
        SSDで予測させる関数。（OpenCVフレームBGR入力）
        戻り値: (rgb_img, predict_bbox, pre_dict_label_index, scores)
        """
        h, w, _ = frame_bgr.shape
        rgb_img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # 前処理（アノテなし）
        phase = "val"
        img_transformed, _, _ = self.transform(frame_bgr, phase, "", "")
        img = torch.from_numpy(img_transformed[:, :, (2, 1, 0)].copy()).permute(2, 0, 1)

        # 推論
        self.net.phase = "inference"
        self.net.eval()
        x = img.unsqueeze(0).to(next(self.net.parameters()).device)
        # with torch.no_grad():
        #     detections = self.net(x)

        with torch.no_grad():
            if x.is_cuda:
                from torch.cuda.amp import autocast
                with autocast():
                    detections = self.net(x)
            else:
                detections = self.net(x)

        if isinstance(detections, tuple):
            detections = detections[0]

        predict_bbox, pre_dict_label_index, scores = [], [], []
        detections = detections.cpu().detach().numpy()
        if detections.ndim == 4:
            find_index = np.where(detections[:, 0:, :, 0] >= data_confidence_level)
            detections_found = detections[find_index]
            for i in range(len(find_index[1])):
                cl = int(find_index[1][i])
                if cl > 0:
                    sc = float(detections_found[i][0])
                    bbox = detections_found[i][1:] * np.array([w, h, w, h], dtype=np.float32)
                    pre_dict_label_index.append(cl - 1)
                    predict_bbox.append(bbox)
                    scores.append(sc)

        return rgb_img, predict_bbox, pre_dict_label_index, scores

    def draw_boxes_cv(self, frame, bbox, label_index, scores, label_names, rgb=True):
        """
        高速なOpenCV描画。frameはRGB/BGRどちらでもOK。戻り値は入力と同じ色空間。
        """
        if len(bbox) == 0:
            return frame
        out = frame.copy()
        for i, bb in enumerate(bbox):
            cls = label_names[label_index[i]]
            sc  = scores[i] if i < len(scores) else None
            txt = f"{cls}:{sc:.2f}" if sc is not None else cls
            x1, y1, x2, y2 = [int(v) for v in bb]
            color = (0, 255, 0)  # シンプルな固定色（高速）
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            # ラベル背景
            t_w = max(60, 10 * len(txt))
            cv2.rectangle(out, (x1, max(0, y1-18)), (x1 + t_w, y1), color, -1)
            cv2.putText(out, txt, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0), 1, cv2.LINE_AA)
        return out
