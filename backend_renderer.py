import torch
import os
import cv2
import numpy as np
import zmq
import argparse
import json
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene import Scene
from eval.openclip_encoder import OpenCLIPNetwork
from gaussian_renderer import render
from utils.vq_utils import get_weights_and_indices

# Helper function for language features
def render_language_feature_map_quick(gaussians, view, pipeline, background, args):
    with torch.no_grad():
        output = render(view, gaussians, pipeline, background, args)
        language_feature_weight_map = output['language_feature_weight_map']
        
        D, H, W = language_feature_weight_map.shape
        
        # Reshape for memory efficiency
        language_feature_weight_map = language_feature_weight_map.view(3, 64, H, W).view(3, 64, H*W)
        
        # Prepare codebooks
        language_codebooks = gaussians._language_feature_codebooks.permute(0, 2, 1)
        
        # EINSUM Operation
        language_feature_map = torch.einsum('ldk,lkn->ldn', language_codebooks, language_feature_weight_map).view(3, 512, H, W)
        
        # Normalization
        language_feature_map = language_feature_map / (language_feature_map.norm(dim=1, keepdim=True) + 1e-10)
        
    return language_feature_map

class BackendRenderer:
    def __init__(self, dataset, pipeline, args):
        self.dataset = dataset
        self.pipeline = pipeline
        self.args = args
        self.device = torch.device("cuda")
        
        # ZMQ Setup
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{args.zmq_port}")
        print(f"🚀 Backend Renderer listening on port {args.zmq_port}")
        
        self.load_models()
        
        # State
        self.current_prompt = ""
        
    def load_models(self):
        print("⏳ Loading Models...")
        self.clip_model = OpenCLIPNetwork(self.device)
        
        self.gaussians = GaussianModel(self.dataset.sh_degree)
        self.dataset.model_path = self.args.ckpt_paths[0]
        self.scene = Scene(self.dataset, self.gaussians, shuffle=False)
        
        # Load Checkpoint
        checkpoint = os.path.join(self.args.ckpt_paths[0], f'chkpnt{self.args.checkpoint}.pth')
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
            
        (model_params, first_iter) = torch.load(checkpoint)
        self.gaussians.restore(model_params, self.args, mode='test')
        
        # Load Language Features
        self.load_language_features()
        
        # Background
        bg_color = [1,1,1] if self.dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        # Get a reference camera for intrinsics
        self.ref_cam = self.scene.getTrainCameras()[0]
        
        print("✅ Models Loaded")

    def load_language_features(self):
        print("⏳ Loading Language Features...")
        language_feature_weights = []
        language_feature_indices = []
        language_feature_codebooks = []
        
        for level_idx in range(3):
            temp_gaussians = GaussianModel(self.dataset.sh_degree)
            ckpt_path = os.path.join(self.args.ckpt_paths[level_idx], f'chkpnt{self.args.checkpoint}.pth')
            (params, _) = torch.load(ckpt_path)
            temp_gaussians.restore(params, self.args, mode='test')
            
            language_feature_codebooks.append(temp_gaussians._language_feature_codebooks.view(-1, 512))
            weights, indices = get_weights_and_indices(temp_gaussians._language_feature_logits, 4)
            language_feature_weights.append(weights)
            language_feature_indices.append(indices + int(level_idx * temp_gaussians._language_feature_codebooks.shape[1]))

        self.gaussians._language_feature_codebooks = torch.stack(language_feature_codebooks, dim=0)
        self.gaussians._language_feature_weights = torch.cat(language_feature_weights, dim=1)
        language_feature_indices = torch.cat(language_feature_indices, dim=1) 
        self.gaussians._language_feature_indices = torch.from_numpy(language_feature_indices.detach().cpu().numpy()).to(self.device)

    def construct_view_camera(self, c2w_matrix, width, height, fov_y):
        # Create a dummy camera object that mimics the structure expected by the renderer
        # We reuse the reference camera but update its extrinsics and intrinsics
        
        view_cam = self.ref_cam
        
        # Update intrinsics
        view_cam.image_width = width
        view_cam.image_height = height
        view_cam.FoVy = fov_y
        view_cam.FoVx = 2 * np.arctan(np.tan(fov_y / 2) * (width / height))
        
        # Update extrinsics
        # c2w_matrix is 4x4 numpy array
        # LangSplat uses World-to-Camera (w2c)
        # w2c = c2w^-1
        
        c2w = torch.from_numpy(c2w_matrix).float().cuda()
        w2c = torch.inverse(c2w)
        
        # The renderer expects world_view_transform to be the transpose of w2c (OpenGL style?)
        # Let's check standard GS implementation.
        # Usually: view_matrix = w2c
        # In GS: world_view_transform = view_matrix.transpose(0, 1)
        
        view_cam.world_view_transform = w2c.transpose(0, 1)
        view_cam.full_proj_transform = (view_cam.world_view_transform.unsqueeze(0).bmm(view_cam.projection_matrix.unsqueeze(0))).squeeze(0)
        view_cam.camera_center = c2w[:3, 3]
        
        return view_cam

    def run(self):
        while True:
            try:
                # Wait for next request from client
                message = self.socket.recv()
                request = json.loads(message)
                
                # Parse request
                c2w = np.array(request['c2w'])
                width = request['width']
                height = request['height']
                fov_y = request['fov_y']
                prompt = request.get('prompt', "")
                threshold = request.get('threshold', 0.22)
                show_heatmap = request.get('show_heatmap', False)
                
                # Update CLIP positives if prompt changed
                if prompt != self.current_prompt and prompt:
                    print(f"🔍 Updating prompt to: {prompt}")
                    self.clip_model.set_positives([prompt])
                    self.current_prompt = prompt
                
                # Construct Camera
                view_cam = self.construct_view_camera(c2w, width, height, fov_y)
                
                # Render RGB
                rgb_out = render(view_cam, self.gaussians, self.pipeline, self.background, self.args)["render"]
                rgb_img = rgb_out.permute(1, 2, 0).detach().cpu().numpy()
                
                final_img = rgb_img
                
                # Render Heatmap if requested
                if show_heatmap and self.current_prompt:
                    lf_map = render_language_feature_map_quick(self.gaussians, view_cam, self.pipeline, self.background, self.args)
                    lf_map = lf_map.permute(0, 2, 3, 1)
                    
                    valid_map = self.clip_model.get_max_across_quick(lf_map)
                    similarity = valid_map.mean(dim=0)[0].cpu().numpy()
                    
                    raw_min = similarity.min()
                    raw_max = similarity.max()
                    
                    if raw_max < threshold:
                        similarity[:] = 0
                    else:
                        similarity = (similarity - raw_min) / (raw_max - raw_min + 1e-8)
                        similarity = similarity ** 4
                    
                    heatmap = cv2.applyColorMap((similarity * 255).astype(np.uint8), cv2.COLORMAP_JET)
                    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
                    
                    final_img = rgb_img * 0.5 + heatmap * 0.5
                
                # Encode image to JPEG
                final_img = (np.clip(final_img, 0, 1) * 255).astype(np.uint8)
                _, buffer = cv2.imencode('.jpg', cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))
                
                # Send reply
                self.socket.send(buffer.tobytes())
                
            except Exception as e:
                print(f"Error processing request: {e}")
                import traceback
                traceback.print_exc()
                self.socket.send(b"ERROR")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--ckpt_root_path", default='output', type=str)
    parser.add_argument("--checkpoint", type=int, default=10000)
    parser.add_argument("--zmq_port", type=int, default=5555)
    
    args = get_combined_args(parser)
    
    # Defaults
    args.sh_degree = 3
    args.white_background = False
    args.language_features_name = "language_features"
    args.images = "images"
    args.resolution = 2
    args.data_device = "cuda"
    args.eval = True
    args.include_feature = True
    args.quick_render = True
    args.compute_cov3D_python = False
    args.convert_SHs_python = False
    args.debug = False
    
    # Paths
    args.dataset_path = f"./data/lerf_ovs/{args.dataset_name}"
    args.source_path = args.dataset_path
    args.ckpt_paths = [
        os.path.join(args.ckpt_root_path, f"{args.dataset_name}_0_{level}") 
        for level in [1, 2, 3]
    ]
    
    renderer = BackendRenderer(model.extract(args), pipeline.extract(args), args)
    renderer.run()
