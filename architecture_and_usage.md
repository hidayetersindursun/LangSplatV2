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

## 🛠️ Troubleshooting
*   **CUDA Out of Memory:** If the backend crashes with OOM, try reducing the resolution in `frontend_viser.py` (currently hardcoded divisor) or restart the backend.
*   **Connection Error:** Ensure the backend is running *before* you try to interact with the frontend.
