# Code to segment an object in a novel view using 3DGS and CLIPSeg

from pathlib import Path

import time
import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from transformers import CLIPProcessor, CLIPSegForImageSegmentation

from PIL import Image

from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager, VanillaDataManagerConfig
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.cameras import camera_utils

# from command import OpenAICommander
from command import OpenAICommander


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
        
        self.times = []

    def get_avg_time(self):
        return np.mean(self.times)

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
            start  = time.time()
            outputs = self.pipeline.model.get_outputs_for_camera(cameras)
            end = time.time()
            self.times.append(end - start)
            
            rgb = outputs["rgb"]
            depth = outputs["depth"]
            
            # convert rgb to numpy
            rgb = rgb.cpu().numpy()
            depth = depth.cpu().numpy()
            if visualize:
                visualize_rgb_depth(rgb, depth)
            return rgb, depth

class Segment3DGS():
    def __init__(self, gs_model: GS):
        print("Loading 3DGS Model...")
        self.gs_model = gs_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # initialize the clip models
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clipseg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(self.device)

    def segment_novel_view(self, x, object_name="chair", visualize=False, segment=False):
        rgb, depth = self.gs_model.get_view(x, visualize=visualize)
        # convert rgb to PIL image
        rgb = Image.fromarray((rgb * 255).astype(np.uint8))
        mask = None
        # Get segmentation
        if segment:
            inputs = self.clip_processor(text=[object_name], images=rgb, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                outputs = self.clipseg_model(**inputs)
            logits_np = outputs.logits.sigmoid().cpu().numpy()
            
            mask = logits_np >= 0.5
            mask = np.array(mask, dtype=np.uint8)
            
            # resize mask to original size
            mask = cv2.resize(mask, (720, 720))
            
            mask = np.array(mask, dtype=np.uint8)
            
            overlay = np.zeros_like(rgb)
            overlay[mask > 0] = (255, 0, 0) 
            
            # Ensure the mask is binary
            alpha = 0.4  # Transparency factor
            if visualize:
                rgb = np.array(rgb)
                output = cv2.addWeighted(overlay, alpha, rgb, 1 - alpha, 0)
                self.visualize_clipseg(rgb, output, object_name)
        
        return rgb, depth, mask
        

    def visualize_clipseg(self, rgb, output, object_name):
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(rgb)
        ax[1].imshow(output)
        # no axis
        ax[0].axis('off')
        ax[1].axis('off')
        # title is object name
        ax[0].set_title('novel view')
        ax[1].set_title(object_name)
        plt.show()
        
if __name__ == '__main__':
    full_phrase = "push the table"
    
    commander = OpenAICommander()
    action, object_ = commander.get_action_and_object(full_phrase)
    
    print("Action: ", action)
    print("Object: ", object_)
    
    
    # splatfacto run
    config_path = 'outputs/panda-data/splatfacto/2024-06-05_151435/config.yml'
    
    gs = GS(config_path)
    
    # get view with novel camera pose
    position = [0.20582463, -0.27465796,  0.87886065]
    quat = [0.43362391, -0.37937454, -0.56434611,  0.59123492]
    gs.get_view((position, quat), visualize=False)
    
    segmenter = Segment3DGS(gs)
    
    # inputs to network are rgb, depth, mask, and
    
    rgb, depth, mask = segmenter.segment_novel_view((position, quat), object_name=object_, visualize=True)
    # plot all three: rgb, depth, mask
    position[1] += 0.5
    
    
    # segment novel view
    objects = ['white object', 'chair', 'table', 'window', 'floor', 'wall']
    for i in range(6):
        object_ = objects[i]
        rgb, depth, mask = segmenter.segment_novel_view((position, quat), object_name=object_, visualize=True, segment=True)
    
    