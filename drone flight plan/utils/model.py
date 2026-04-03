import torch
from utils.ssd_model import SSD
from utils.ssd_predict_show import SSDPredictShow


class Model:

    def __init__(self, weight_path: str = "./weights/ssd_finetuned_200_filter.pth", use_cuda: bool = True):
        # 使用デバイス
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and use_cuda) else "cpu")
        print(f"[MODEL] Using device: {self.device}")
        if torch.cuda.is_available() and use_cuda:
            torch.backends.cudnn.benchmark = True

        # VOC20クラス（背景は別扱いで+1）
        self.voc_classes = (
            # "apple", "orange", "banana", "others"
            "apple", "orange", "banana"
        )

        # SSD設定
        self.ssd_cfg = {
            'num_classes': len(self.voc_classes) + 1,
            'input_size': 300,
            'bbox_aspect_num': [4, 6, 6, 6, 4, 4],
            'feature_maps': [38, 19, 10, 5, 3, 1],
            'steps': [8, 16, 32, 64, 100, 300],
            'min_sizes': [21, 45, 99, 153, 207, 261],
            'max_sizes': [45, 99, 153, 207, 261, 315],
            'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        }

        # ===== ネット構築＆重みロード =====
        # self.netb = SSD(phase="inference", cfg=self.ssd_cfg)
        # self.netb.to(self.device)
        # self.netb.eval()         # netb = SSD(phase="inference", cfg=ssd_cfg)
        # net_weights_base = torch.load(                                   # net_weights_base = torch.load(...)
        #     weight_path, map_location={'cuda:0': 'cpu'}
        # )
        # # そのままロード（キー相違への寛容性を少し上げたい場合は strict=False）
        # self.netb.load_state_dict(net_weights_base)                      # netb.load_state_dict(net_weights_base)

        self.netb = SSD(phase="inference", cfg=self.ssd_cfg)
        state = torch.load(weight_path, map_location='cpu')
        missing, unexpected = self.netb.load_state_dict(state, strict=False)
        if unexpected:
            print(f"[MODEL] unexpected keys ignored: {unexpected[:6]}{'...' if len(unexpected)>6 else ''}")
        if missing:
            print(f"[MODEL] missing keys: {missing[:6]}{'...' if len(missing)>6 else ''}")
        self.netb.to(self.device).eval()
        # GPU最適化（任意）
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 300, 300, device=self.device)
                _ = self.netb(dummy)  # warmup
        print(f"[MODEL] Loaded weights from {weight_path}")

        # 推論・表示ユーティリティ
        self.ssdf = SSDPredictShow(eval_categories=self.voc_classes, net=self.netb)

        
        print("[CHK] net device:", next(self.netb.parameters()).device)
        print("[CHK] dbox device:", self.netb.dbox_list.device)


    # ---------- 画像ファイルで推論 ----------
    def predict(self, image_path: str, conf: float = 0.5):
        """画像ファイルに対する推論（結果を返す）"""
        return self.ssdf.ssd_predict(image_path, data_confidence_level=conf)

    def show(self, image_path: str, conf: float = 0.5):
        """画像ファイルに対する推論（可視化表示）"""
        self.ssdf.show(image_path, data_confidence_level=conf)           # ssdf.show(image_path, ...)

    # ---------- フレーム(BGR)で推論 ----------
    def predict_frame(self, frame_bgr, conf: float = 0.5):
        """OpenCVのフレーム(BGR)に対する推論（結果を返す）"""
        return self.ssdf.ssd_predict_from_frame(frame_bgr, data_confidence_level=conf)

    def annotate_frame(self, frame_bgr, conf: float = 0.5):
        """
        OpenCVフレーム(BGR)に検出結果を重畳してBGRで返す（cv2.imshowにそのまま出せる）
        """
        rgb_img, boxes, labels, scores = self.predict_frame(frame_bgr, conf=conf)
        out_bgr = self.ssdf.draw_boxes_cv(frame_bgr, boxes, labels, scores, self.voc_classes, rgb=False)
        return out_bgr
