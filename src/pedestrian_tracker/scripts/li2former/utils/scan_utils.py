'''
@file: scan_utils.py
@breif: the scan tools functions
@author: Winter
@update: 2023.9.30
'''
import numpy as np
import math


def rphi2xy(r: np.ndarray, phi: np.ndarray):
    '''
    Conversion from polar coordinates to Cartesian coordinates.
    '''
    return r * np.cos(phi), r * np.sin(phi)

def xy2rphi(x, y):
    '''
    Conversion from Cartesian coordinates to polar coordinates.
    '''
    return np.hypot(x, y), np.arctan2(y, x)

def global2local(scan_r: np.ndarray, scan_phi: np.ndarray, global_r: np.ndarray, global_phi: np.ndarray):
    '''
    Transforming laser points from a global polar coordinate system to a local cartesian coordinate system.

    @NOTE: As the 2D range data is inherently rotation invariant, learning offsets (∆x, ∆y) locally
    is better than absolute (x, y). Each (∆x, ∆y) is aligned at the current laser point window.
    Origin at the scan points (r, phi), y forward along scan ray, x rightward. 

    Parameters
    ----------
    scan_r: np.ndarray
        laser distance information.
    scan_phi: np.ndarray
        laser angle information.
    global_r: np.ndarray
        distance information for detecting targets (global).
    global_phi: np.ndarray
        angle information for detecting targets (global).
    
    Return
    ----------
    dx: np.ndarray
        the x-axis value of the detected target in the local coordinate system.
    dy: np.ndarray
        the y-axis value of the detected target in the local coordinate system.
    '''
    dx = np.sin(global_phi - scan_phi) * global_r
    dy = np.cos(global_phi - scan_phi) * global_r - scan_r
    return dx, dy

def local2global(scan_r: np.ndarray, scan_phi: np.ndarray, dx: np.ndarray, dy: np.ndarray, cartesian: bool=False):
    '''
    Transforming laser points from a local cartesian coordinate system to a global polar coordinate system.

    @NOTE: As the 2D range data is inherently rotation invariant, learning offsets (∆x, ∆y) locally
    is better than absolute (x, y). Each (∆x, ∆y) is aligned at the current laser point window.
    Origin at the scan points (r, phi), y forward along scan ray, x rightward. 

    Parameters
    ----------
    scan_r: np.ndarray
        laser distance information.
    scan_phi: np.ndarray
        laser angle information.
    dx: np.ndarray
        the x-axis value of the detected target in the local coordinate system.
    dy: np.ndarray
        the y-axis value of the detected target in the local coordinate system.
    cartesian: bool
        return value is in cartesian coordinate system if true else polar system.
    
    Return
    ----------
    global_r / global_x: np.ndarray
        distance information or x-axis value for detecting targets (global).
    global_phi / global_y: np.ndarray
        angle information or y-axis value for detecting targets (global).
    '''
    tmp_y   = scan_r + dy
    tmp_phi = np.arctan2(dx, tmp_y)
    global_phi, global_r = tmp_phi + scan_phi, tmp_y / np.cos(tmp_phi)
    if cartesian:
        return rphi2xy(global_r, global_phi)
    return global_r, global_phi

def local2box(scan_r: np.ndarray, scan_phi: np.ndarray, dx: np.ndarray, dy: np.ndarray, w: float, h: float):
    x, y = local2global(scan_r, scan_phi, dx, dy, cartesian=True)
    return np.hstack([x[:, None] - 0.5 * w, y[:, None] - 0.5 * h, x[:, None] + 0.5 * w, y[:, None] + 0.5 * h])

def augmentation(sample_dict):
    scans, target_reg = sample_dict["scans"], sample_dict["target_reg"]

    # # Random scaling
    # s = np.random.uniform(low=0.95, high=1.05)
    # scans = s * scans
    # target_reg = s * target_reg

    # Random left-right flip. Of whole batch for convenience, but should be the
    # same as individuals.
    if np.random.rand() < 0.5:
        scans = scans[:, ::-1]
        target_reg[:, 0] = -target_reg[:, 0]

    sample_dict.update({"target_reg": target_reg, "scans": scans})

    return sample_dict

def scans2cutout(scans: np.ndarray, scan_phi: np.ndarray, stride: int=1, centered: bool=True,
    fixed: bool=False, window_width: float=1.66, window_depth: float=1.0, num_cutout_pts: int=48,
    padding_val: float=29.99, area_mode: bool=False) -> np.ndarray:
    '''
    Cut out the scans.

    Parameters
    ----------
    scans: np.ndarray   (N, n_pts)
        the scans matrix, N is the frame number and n_pts is the lidar points within a frame
    scan_phi: np.ndarray
        [-fov_phi / 2, fov_phi / 2]
    stride: int
    centered: bool
        center and normalize the lidar cut-out if true else do not
    fixed: bool
    window_width, window_depth: float
        the width and depth of the cut-out window
    num_cutout_pts: int
        the number of cutout points
    padding_val: float
        the padding value of the outbound area
    area_mode: bool
        use the area sampling if true else do not

    Return
    ----------
    cutout: np.ndarray
        the cutout of scans (scans, times, cutouts)
    '''        
    num_scans, num_pts = scans.shape

    # size (width) of the window
    dists = (
        scans[:, ::stride]
        if fixed
        else np.tile(scans[-1, ::stride], num_scans).reshape(num_scans, -1)
    )
    half_alpha = np.arctan(0.5 * window_width / np.maximum(dists, 1e-2))
    
    # cutout indices
    delta_alpha = 2.0 * half_alpha / (num_cutout_pts - 1)

    # For each ray direction, define a window and divide the opening of the window equally
    # into `num_cutout_pts`` parts to obtain the angle for each part.
    ang_ct = (
        scan_phi[::stride]
        - half_alpha
        + np.arange(num_cutout_pts).reshape(num_cutout_pts, 1, 1) * delta_alpha
    )
    ang_ct = (ang_ct + np.pi) % (2.0 * np.pi) - np.pi  # warp angle

    # Starting from scan_phi[0], the number of increment to reach `ang_ct`` divisions.
    inds_ct = (ang_ct - scan_phi[0]) / (scan_phi[1] - scan_phi[0])
    outbound_mask = np.logical_or(inds_ct < 0, inds_ct > num_pts - 1)

    # cutout (linear interp)
    inds_ct_low = np.clip(np.floor(inds_ct), 0, num_pts - 1).astype(np.int64)
    inds_ct_high = np.clip(inds_ct_low + 1, 0, num_pts - 1).astype(np.int64)
    inds_ct_ratio = np.clip(inds_ct - inds_ct_low, 0.0, 1.0)
    inds_offset = (np.arange(num_scans).reshape(1, num_scans, 1) * num_pts) 

    ct_low = np.take(scans, inds_ct_low + inds_offset)
    ct_high = np.take(scans, inds_ct_high + inds_offset)

    # Due to the angles divided within the window not aligning exactly with the laser beams,
    # direct correspondence to laser distance is not possible. However, this angle is definitely
    # situated between two laser beams (`ct_low` and `ct_high``), and a linear weighting is applied
    # to this ranging measurement (higher weight given to the beam it is closer to).
    ct = ct_low + inds_ct_ratio * (ct_high - ct_low)

    # use area sampling for down-sampling (close points)
    if area_mode:
        num_pts_in_window = inds_ct[-1] - inds_ct[0]
        area_mask = num_pts_in_window > num_cutout_pts
        if np.sum(area_mask) > 0:
            # sample the window with more points than the actual number of points
            s_area = int(math.ceil(np.max(num_pts_in_window) / num_cutout_pts))
            num_ct_pts_area = s_area * num_cutout_pts
            delta_alpha_area = 2.0 * half_alpha / (num_ct_pts_area - 1)
            ang_ct_area = (
                scan_phi[::stride]
                - half_alpha
                + np.arange(num_ct_pts_area).reshape(num_ct_pts_area, 1, 1)
                * delta_alpha_area
            )
            ang_ct_area = (ang_ct_area + np.pi) % (2.0 * np.pi) - np.pi  # warp angle
            inds_ct_area = (ang_ct_area - scan_phi[0]) / (scan_phi[1] - scan_phi[0])
            inds_ct_area = np.rint(np.clip(inds_ct_area, 0, num_pts - 1)).astype(np.int32)
            ct_area = np.take(scans, inds_ct_area + inds_offset)
            ct_area = ct_area.reshape(
                num_cutout_pts, s_area, num_scans, dists.shape[1]
            ).mean(axis=1)
            ct[:, area_mask] = ct_area[:, area_mask]

    # normalize cutout
    ct[outbound_mask] = padding_val
    ct = np.clip(ct, dists - window_depth, dists + window_depth)
    if centered:
        ct = ct - dists
        ct = ct / window_depth

    return np.ascontiguousarray(ct.transpose((2, 1, 0)), dtype=np.float32) 