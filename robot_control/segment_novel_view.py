# Segment out a novel view from a given image from GS.


from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager, VanillaDataManagerConfig
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.cameras import camera_utils


CAMERA_CONFIG = {
    'fx': 584.1088,
    'fy': 584.1088,
    'cx': 360,
    'cy': 360,
    'width': 720,
    'height': 720,
    'camera_type': 1,
}

DIS_PARAMS = torch.tensor([0., 0., 0., 0., 0., 0.])

SCALE_POSES = 1.9018828363660438
MEAN_ORIGIN = [0.2530, 0.0189, 0.8645]
MEAN_UP = [0.3894, 0.0702, 0.8461]

CAM_TYPE = CameraType.PERSPECTIVE

def visualize_rgb_depth(rgb, depth):
    # plot rgb and depth in the same figure
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(rgb)
    ax[1].imshow(depth)
    plt.show()


class GS():
    def __init__(self, config_path):
        self.config_path = Path(config_path)
        
        # initialize the model
        self.model = None
        self.data = None
        self.downscale_factor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        def update_config(config: TrainerConfig) -> TrainerConfig:
            data_manager_config = config.pipeline.datamanager
            assert isinstance(data_manager_config, (VanillaDataManagerConfig, FullImageDatamanagerConfig))
            data_manager_config.eval_num_images_to_sample_from = -1
            data_manager_config.eval_num_times_to_repeat_images = -1
            if isinstance(data_manager_config, VanillaDataManagerConfig):
                data_manager_config.train_num_images_to_sample_from = -1
                data_manager_config.train_num_times_to_repeat_images = -1
            if self.data is not None:
                data_manager_config.data = self.data
            if self.downscale_factor is not None:
                assert hasattr(data_manager_config.dataparser, "downscale_factor")
                setattr(data_manager_config.dataparser, "downscale_factor", self.downscale_factor)
            return config
        
        self.config, self.pipeline, _, _ = eval_setup(
            self.config_path,
            eval_num_rays_per_chunk=None,
            test_mode="inference",
            update_config_callback=update_config
        )
        self.cx = torch.tensor([[CAMERA_CONFIG['cx']]]).to(self.device)
        self.cy = torch.tensor([[CAMERA_CONFIG['cy']]]).to(self.device)
        self.fx = torch.tensor([[CAMERA_CONFIG['fx']]]).to(self.device)
        self.fy = torch.tensor([[CAMERA_CONFIG['fy']]]).to(self.device)
        self.height = torch.tensor([[CAMERA_CONFIG['height']]]).to(self.device)
        self.width = torch.tensor([[CAMERA_CONFIG['width']]]).to(self.device)

    def get_view(self, x, visualize=False):
        """_summary_

        Args:
            x (tuple): position and orientation (quat)
        """
        position, orientation = x
        x, y, z, w = orientation
        
        # convert orientation to rotation matrix
        quat = R.from_quat([x, y, z, w])
        rot = quat.as_matrix()
        
        # create transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = rot
        transform[:3, 3] = position
        # set last row to [0, 0, 0, 1]
        transform[3, :] = np.array([0, 0, 0, 1])
        
        pose = torch.from_numpy(np.array(transform).astype(np.float32))
        # add extra dim for batch
        pose = pose.unsqueeze(0)
        
        mean_origin = torch.tensor(MEAN_ORIGIN)
        mean_up = torch.tensor(MEAN_UP)
        
        poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
            pose,
            method="up",
            center_method="poses",
            mean_origin=mean_origin,
            mean_up=mean_up,
        )
        
        poses[:, :3, 3] *= SCALE_POSES
        camera_to_worlds = poses[:, :3, :4]
        cameras = Cameras(
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            distortion_params=DIS_PARAMS,
            height=self.height,
            width=self.width,
            camera_to_worlds=camera_to_worlds,
            camera_type=CAM_TYPE,
            metadata={},
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.pipeline.model.get_outputs_for_camera(cameras)
            rgb = outputs["rgb"]
            depth = outputs["depth"]
            if visualize:
                # convert rgb to numpy
                rgb = rgb.cpu().numpy()
                depth = depth.cpu().numpy()
                # plot rgb and depth in the same figure
                visualize_rgb_depth(rgb, depth)
            return rgb, depth

class Segment3DGS():
    def __init__(self, gs_model: GS):
        self.gs_model = gs_model
        
        # initialize the model

    def segment_novel_view(self, x, object):
        rgb, depth = self.gs_model.get_view(x)


if __name__ == '__main__':
    # best depth 
    config_path = 'outputs/panda-data/splatfacto/2024-06-05_151435/config.yml'
    
    gs = GS(config_path)
    
    # get view with novel camera pose
    position = [0.20582463, -0.27465796,  0.87886065]
    quat = [0.43362391, -0.37937454, -0.56434611,  0.59123492]
    gs.get_view((position, quat), visualize=True)
    
    print('Segmenting novel view...')