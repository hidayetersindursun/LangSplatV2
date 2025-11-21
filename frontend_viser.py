import viser
import zmq
import numpy as np
import cv2
import time
import argparse
import json

class ViserFrontend:
    def __init__(self, args):
        self.args = args
        
        # ZMQ Setup
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://localhost:{args.zmq_port}")
        print(f"🔗 Connected to backend on port {args.zmq_port}")
        
        # Viser Server
        self.server = viser.ViserServer(port=args.port)
        print(f"Viser server started at http://localhost:{args.port}")
        
        # State
        self.current_prompt = "chair"
        self.last_camera_pose = None
        self.need_update = True
        self.waiting_for_reply = False
        
        # GUI Elements
        self.setup_gui()
        
        # Start Render Loop
        self.render_loop()
        
    def setup_gui(self):
        with self.server.gui.add_folder("LangSplat Controls"):
            self.prompt_input = self.server.gui.add_text("Prompt", initial_value=self.current_prompt)
            self.search_button = self.server.gui.add_button("Search")
            self.threshold_slider = self.server.gui.add_slider("Threshold", min=0.0, max=1.0, step=0.01, initial_value=0.22)
            self.show_heatmap_checkbox = self.server.gui.add_checkbox("Show Heatmap", initial_value=True)
            self.resolution_slider = self.server.gui.add_slider("Resolution Divisor", min=1, max=8, step=1, initial_value=2)
            self.reset_camera_button = self.server.gui.add_button("Reset Camera")
            
        self.search_button.on_click(self.update_prompt)
        self.threshold_slider.on_update(lambda _: setattr(self, 'need_update', True))
        self.show_heatmap_checkbox.on_update(lambda _: setattr(self, 'need_update', True))
        self.reset_camera_button.on_click(self.reset_camera)
        
        # Register client connect handler
        self.server.on_client_connect(self.handle_new_client)
        
    def update_prompt(self, _):
        new_prompt = self.prompt_input.value
        if new_prompt != self.current_prompt:
            print(f"🔍 Updating prompt to: {new_prompt}")
            self.current_prompt = new_prompt
            self.need_update = True
            
    def handle_new_client(self, client: viser.ClientHandle):
        print("New client connected!")
        # We can't easily reset camera to training view without backend info, 
        # but we can set a reasonable default or wait for first render.
        
    def reset_camera(self, _):
        # This would ideally request the initial camera pose from backend
        # For now, just a placeholder or we can implement a specific request type
        pass

    def render_loop(self):
        while True:
            clients = self.server.get_clients()
            if not clients:
                time.sleep(0.1)
                continue
                
            client = list(clients.values())[0]
            
            # Check if camera moved
            current_position = client.camera.position
            current_wxyz = client.camera.wxyz
            
            if self.last_camera_pose is None:
                self.need_update = True
                self.last_camera_pose = (current_position, current_wxyz)
            else:
                last_position, last_wxyz = self.last_camera_pose
                if not np.allclose(current_position, last_position) or not np.allclose(current_wxyz, last_wxyz):
                    self.need_update = True
                    self.last_camera_pose = (current_position, current_wxyz)
            
            if self.need_update and not self.waiting_for_reply:
                self.send_render_request(client)
            
            time.sleep(0.01)
            
    def send_render_request(self, client):
        try:
            self.waiting_for_reply = True
            
            # Get Camera Pose (Viser is C2W)
            position = client.camera.position
            wxyz = client.camera.wxyz
            
            # Convert wxyz to rotation matrix
            # R = ... (using scipy or manual)
            # Manual conversion for dependency minimization
            w, x, y, z = wxyz
            R = np.array([
                [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
                [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
                [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
            ])
            
            c2w = np.eye(4)
            c2w[:3, :3] = R
            c2w[:3, 3] = position
            
            # Get viewport size
            # Note: Viser doesn't explicitly send viewport size in client object easily in all versions
            # We'll assume a default or make it configurable. 
            # Actually, for rendering we need an aspect ratio.
            # Let's use a fixed resolution scaled by divisor for now.
            W = 1280 // self.resolution_slider.value
            H = 720 // self.resolution_slider.value
            
            fov_y = client.camera.fov
            
            request = {
                'c2w': c2w.tolist(),
                'width': int(W),
                'height': int(H),
                'fov_y': fov_y,
                'prompt': self.current_prompt,
                'threshold': self.threshold_slider.value,
                'show_heatmap': self.show_heatmap_checkbox.value
            }
            
            self.socket.send_json(request)
            
            # Receive Image
            message = self.socket.recv()
            
            if message == b"ERROR":
                print("Backend returned error")
                self.waiting_for_reply = False
                return
                
            # Decode Image
            nparr = np.frombuffer(message, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Use new API for background image
            client.scene.set_background_image(img)
            
            self.waiting_for_reply = False
            self.need_update = False
            
        except Exception as e:
            print(f"Communication error: {e}")
            self.waiting_for_reply = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8081, help="Viser server port")
    parser.add_argument("--zmq_port", type=int, default=5555, help="ZMQ port")
    args = parser.parse_args()
    
    frontend = ViserFrontend(args)
