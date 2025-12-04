# LangSplat Viser Viewer - Architecture & Usage

This project uses a **Client-Server Architecture** to bridge the gap between the legacy LangSplat environment (Python 3.7) and the modern Viser GUI library (Python 3.10+).

## 🏗️ Architecture

We split the application into two separate processes that communicate via **ZeroMQ**:

1.  **Backend (`backend_renderer.py`)**
    *   **Environment:** `langsplat_v2` (Python 3.7, CUDA 11.6)
    *   **Role:** Loads the heavy Gaussian Splatting models and CLIP. Handles rendering requests.
    *   **Communication:** Listens on TCP port `5555`.

2.  **Frontend (`frontend_viser.py`)**
    *   **Environment:** `viser_env` (Python 3.10+)
    *   **Role:** Runs the Viser web server and GUI. Captures user input (camera, prompts).
    *   **Communication:** Sends requests to the backend and displays the returned images.

## 🚀 How to Run

You need **two separate terminal windows** to run the viewer.

### Terminal 1: Backend (Renderer)
This process does the heavy lifting.

```bash
# Activate the LangSplat environment
conda activate langsplat_v2

# Run the renderer (replace 'waldo_kitchen' with your dataset)
python backend_renderer.py --dataset_name waldo_kitchen
```
*Wait until you see `🚀 Backend Renderer listening on port 5555`.*

### Terminal 2: Frontend (Viewer)
This process runs the web interface.

```bash
# Activate the Viser environment
conda activate viser_env

# Run the viewer
python frontend_viser.py
```

### 🖥️ Open in Browser
Once both are running, open your browser and go to:
**http://localhost:8081**

## 🎮 Controls
*   **Navigation:** standard Viser controls (Left Click to rotate, Right Click to pan, Scroll to zoom).
*   **Prompt:** Enter a text query (e.g., "apple", "chair") and click **Search**.
*   **Threshold:** Adjust the slider to filter the heatmap visibility.
*   **Show Heatmap:** Toggle the heatmap overlay on/off.
*   **Reset Camera:** Moves the camera back to the first training position.

## 👁️ Simple Viser Viewer (RGB Only)

If you just want to visualize the trained geometry (RGB only) without language features, you can use the lightweight `simple_viser.py`.

*   **Environment:** `viser_env`
*   **Usage:**
    ```bash
    conda activate viser_env
    python simple_viser.py --ply_path output/your_model/point_cloud/iteration_30000/point_cloud.ply
    ```

## 🚗 Car Dataset Workflow (Current Progress)

This section documents the specific workflow used for the "Araba" (Car) dataset, which had pre-existing COLMAP data.

### 1. RGB Training (Geometry)
First, we trained the standard Gaussian Splatting model to establish the scene geometry.
*   **Command:**
    ```bash
    conda activate langsplat_v2
    python train.py -s /path/to/dataset/colmap -m output/araba_test
    ```
*   **Outcome:** A `.ply` file containing the 3D Gaussians (RGB only).
*   **Verification:** We used `simple_viser.py` to inspect the geometry and confirmed it looks good.

### 2. Feature Extraction (Preprocessing)
Now, we extract language features (CLIP embeddings) from the dataset images.
*   **Command:**
    ```bash
    conda activate langsplat_v2
    python preprocess.py --dataset_path /path/to/dataset/colmap
    ```
*   **What it does:**
    *   Generates masks using SAM (Segment Anything Model).
    *   Extracts CLIP features for the masked regions.
    *   Saves these features to a `language_features` folder within the dataset.

### 3. Feature Training (Language Field)
*Next Step:* Train the language field on top of the frozen RGB model.
*   **Command:**
    ```bash
    conda activate langsplat_v2
    python train.py -s /path/to/dataset/colmap -m output/araba_feature \
      --include_feature --start_checkpoint output/araba_test/chkpnt30000.pth
    ```

### 4. Interactive Visualization
*Final Step:* Use the full Client-Server architecture to query the scene.
*   **Backend:** `python backend_renderer.py --dataset_name araba_feature`
*   **Frontend:** `python frontend_viser.py`

## 🛠️ Troubleshooting
*   **CUDA Out of Memory:** If the backend crashes with OOM, try reducing the resolution in `frontend_viser.py` (currently hardcoded divisor) or restart the backend.
*   **Connection Error:** Ensure the backend is running *before* you try to interact with the frontend.
