"""
@file: detector.py
@breif: the 2d person detector wrapper for ROS etc.
@author: Winter
@update: 2023.12.11
"""
import torch
import numpy as np
from collections import deque
from .utils import scan_utils as scan_u
from .utils import eval_utils as eval_u
from .models import buildModel


class Detector:
    def __init__(self, ros_config, ckpt_file) -> None:
        self.config = ros_config

        ckpt = torch.load(ckpt_file)
        self.detector = buildModel(ros_config)
        self.detector.load_state_dict(ckpt["model_state"])
        self.detector.eval()

        self.gpu = ros_config("MODEL")["GPU"]

        if self.gpu:
            self.detector.cuda()

        self.scan_phi = None
        self.scan_stride = ros_config("SCAN_STRIDE")
        self.laser_fov = None
        self.history = deque(maxlen=ros_config("NUM_SCANS"))

    def __str__(self) -> str:
        return "Detector for Pedestrians."

    def __call__(self, scan):
        assert self.ready(), "Call setLaserFov() first."

        if self.scan_phi is None:
            half_fov_rad = 0.5 * np.deg2rad(self.laser_fov)
            self.scan_phi = np.linspace(
                -half_fov_rad, half_fov_rad, len(scan), dtype=np.float32
            )


        cutout_kwargs = {k.lower(): v for k, v in self.config("CUTOUT_KWARGS").items()}
        cutout = scan_u.scans2cutout(
            scan[None, ...],
            self.scan_phi,
            stride=self.config("POINT_STRIDE"),
            **cutout_kwargs
        )
        self.history.append(cutout)
        ct = np.concatenate(self.history, axis=1)
        ct = torch.from_numpy(ct).float()

        if self.gpu:
            ct = ct.cuda()

        # inference
        with torch.no_grad():
            # one extra dimension for batch
            pred_cls, pred_reg = self.detector(ct.unsqueeze(dim=0))

        pred_cls = torch.sigmoid(pred_cls[0]).data.cpu().numpy()
        pred_reg = pred_reg[0].data.cpu().numpy()

        # Non-Maximum Suppression
        dets_xy, dets_cls, _ = eval_u.nms(
            scan[::self.scan_stride][None, :],
            self.scan_phi[::self.scan_stride],
            pred_cls[:, 0],
            pred_reg
        )

        return dets_xy, dets_cls


    def ready(self):
        return self.laser_fov is not None

    def setLaserFov(self, fov: float):
        self.laser_fov = fov


# import torch
# import numpy as np

# from dr_spaam.model.drow_net import DrowNet
# from dr_spaam.model.dr_spaam import DrSpaam
# from dr_spaam.utils import utils as u


# class Detector(object):
#     def __init__(
#         self, ckpt_file, model="DROW3", gpu=True, stride=1, panoramic_scan=False
#     ):
#         """A warpper class around DROW3 or DR-SPAAM network for end-to-end inference.

#         Args:
#             ckpt_file (str): Path to checkpoint
#             model (str): Model name, "DROW3" or "DR-SPAAM".
#             gpu (bool): True to use GPU. Defaults to True.
#             stride (int): Downsample scans for faster inference.
#             panoramic_scan (bool): True if the scan covers 360 degree.
#         """
#         self._gpu = gpu
#         self._stride = stride
#         self._use_dr_spaam = model == "DR-SPAAM"

#         self._scan_phi = None
#         self._laser_fov_deg = None

#         if model == "DROW3":
#             self._model = DrowNet(
#                 dropout=0.5, cls_loss=None, mixup_alpha=0.0, mixup_w=0.0
#             )
#         elif model == "DR-SPAAM":
#             self._model = DrSpaam(
#                 dropout=0.5,
#                 num_pts=56,
#                 embedding_length=128,
#                 alpha=0.5,
#                 window_size=17,
#                 panoramic_scan=panoramic_scan,
#                 cls_loss=None,
#                 mixup_alpha=0.0,
#                 mixup_w=0.0,
#             )
#         else:
#             raise NotImplementedError(
#                 "model should be 'DROW3' or 'DR-SPAAM', received {} instead.".format(
#                     model
#                 )
#             )

#         ckpt = torch.load(ckpt_file)
#         self._model.load_state_dict(ckpt["model_state"])

#         self._model.eval()
#         if gpu:
#             torch.backends.cudnn.benchmark = True
#             self._model = self._model.cuda()

#     def __call__(self, scan):
#         if self._scan_phi is None:
#             assert self.is_ready(), "Call set_laser_fov() first."
#             half_fov_rad = 0.5 * np.deg2rad(self._laser_fov_deg)
#             self._scan_phi = np.linspace(
#                 -half_fov_rad, half_fov_rad, len(scan), dtype=np.float32
#             )

#         # preprocess
#         ct = u.scans_to_cutout(
#             scan[None, ...],
#             self._scan_phi,
#             stride=self._stride,
#             centered=True,
#             fixed=True,
#             window_width=1.0,
#             window_depth=0.5,
#             num_cutout_pts=56,
#             padding_val=29.99,
#             area_mode=True,
#         )
#         ct = torch.from_numpy(ct).float()

#         if self._gpu:
#             ct = ct.cuda()

#         # inference
#         with torch.no_grad():
#             # one extra dimension for batch
#             if self._use_dr_spaam:
#                 pred_cls, pred_reg, _ = self._model(ct.unsqueeze(dim=0), inference=True)
#             else:
#                 pred_cls, pred_reg = self._model(ct.unsqueeze(dim=0))

#         pred_cls = torch.sigmoid(pred_cls[0]).data.cpu().numpy()
#         pred_reg = pred_reg[0].data.cpu().numpy()

#         # postprocess
#         dets_xy, dets_cls, instance_mask = u.nms_predicted_center(
#             scan[:: self._stride],
#             self._scan_phi[:: self._stride],
#             pred_cls[:, 0],
#             pred_reg,
#         )

#         return dets_xy, dets_cls, instance_mask

#     def set_laser_fov(self, fov_deg):
#         self._laser_fov_deg = fov_deg

#     def is_ready(self):
#         return self._laser_fov_deg is not None
