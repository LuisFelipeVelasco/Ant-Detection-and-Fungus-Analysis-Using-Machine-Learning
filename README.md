# 🐜 Ant Detection and Fungus Analysis Using Machine Learning

> **Applying Computer Vision and clustering algorithms — built from scratch — to detect, individualize, and track ants in both still images and live video, with optional fungus interaction analysis.**

---
https://github.com/user-attachments/assets/190f7413-f283-4a63-b8aa-6a04f90cb52c
## 🎯 Purpose

Ants are small, fast, and visually similar to background elements, making automated detection a genuine Computer Vision challenge. This project tackles two related problems:

- 📸 **Image Analysis** (`Ant_Fungus_Interaction_Analysis.ipynb`) — Given a single photograph, detect all ants present, individualize them as separate clusters, and compute the percentage of each ant's body that overlaps with a fungus region.
- 🎥 **Video Tracking** (`Ant_Tracker.py`) — Given a video, detect all ants in the first frame, let the user select one to follow, and continuously track its trajectory across the entire video — optionally detecting when it passes over a fungus.

Both components share the same core idea: use color-based pixel filtering to isolate ant-colored regions, then apply clustering to separate individual ants from one another.

---

## 🗂️ Project Structure

```
Ant-Detection-and-Fungus-Analysis/
│
├── Ant_Fungus_Interaction_Analysis.ipynb   # 📓 Image analysis pipeline (K-Means + DBSCAN from scratch)
│
└── Ant_Tracker.py                          # 🎥 Video tracking pipeline (DBSCAN + OpenCV)
```

---

## 📓 Ant Fungus Interaction Analysis — Image Notebook

### What It Does

Given a photograph of ants on a surface, the notebook:

1. **Detects ant pixels** by filtering for reddish-brown tones in RGB space.
2. **Clusters the pixels** into individual ants using both **K-Means** and **DBSCAN**, both implemented from scratch in pure Python/NumPy.
3. **Detects the fungus region** by applying a separate color filter and filling enclosed interior gaps.
4. **Computes overlap** — for each detected ant, reports the percentage of its body that lies over the fungus.

> 💡 **Paste a result image here showing the detected ants with their cluster labels.**

> 🎬 *Example output: ants shown as colored point clouds over the grey fungus region, each labeled with its fungus overlap percentage.*

### 🧠 Algorithms Implemented From Scratch

Both clustering algorithms are written entirely in Python without using scikit-learn or any ML library:

#### K-Means
| Step | Description |
|---|---|
| 1 | Initialize K centroids at random positions within the image bounds |
| 2 | Assign each ant pixel to its nearest centroid using squared Euclidean distance |
| 3 | Update each centroid to the mean position of all its assigned pixels |
| 4 | Repeat steps 2–3 until centroids stop moving (convergence) |

#### DBSCAN
| Step | Description |
|---|---|
| 1 | Classify each point as **core** (≥ `min_points` neighbors within `ε`) or **noise** |
| 2 | Upgrade noise points that fall within `ε` of a core point to **border** points |
| 3 | Expand clusters from each unvisited core point by absorbing neighboring core/border points |
| 4 | Discard small clusters whose size falls below `b × average_cluster_size` |
| 5 | Renumber class labels to be consecutive after any discarded clusters |

### Pipeline

```
Input Image
    │
    ├──► Color Filter (RGB thresholds) → Ant pixel mask
    │        │
    │        ├──► K-Means (from scratch) → Cluster per ant
    │        └──► DBSCAN (from scratch)  → Cluster per ant (more robust)
    │                 │
    │                 └──► Noise removal → Small cluster pruning → Relabeling
    │
    ├──► Color Filter (RGB thresholds) → Fungus pixel mask
    │        │
    │        └──► Gap filling (flood fill via accumulate) → Solid fungus region
    │
    └──► Overlap Analysis → % of each ant over the fungus → Visualization
```

---

## 🎥 Ant Tracker — Video Script

### What It Does

1. **Opens** a video file (`ant_video.mp4`) and reads the first frame.
2. **Detects all ants** in that frame using color filtering + DBSCAN, draws labeled bounding boxes.
3. **Prompts the user** to select one ant label to follow.
4. **Tracks the chosen ant** across every subsequent frame using a sliding ROI window that follows the ant's movement.
5. **Optionally detects fungus** by extracting a background image (median of random frames) and checking whether the ant's centroid lands inside the fungus region each frame.
6. **Plots the full trajectory** at the end, with start (green) and end (red) points marked, and orange/blue coloring to indicate on/off fungus if enabled.

> 💡 **Paste a video here showing the chosen ant and its travel path drawn on the plane.**

### Key Design Decisions

**ROI Masking** — Rather than processing the full frame every tick, a rectangular window follows the tracked ant. This drastically reduces the pixel search space and eliminates detections of other ants in distant parts of the frame.

**Background Extraction** — The fungus region is isolated by computing the pixel-wise **median** of 100 randomly sampled frames. Moving objects (ants) cancel out in the median, leaving only the static background with the fungus clearly visible.

**Proximity-Based Centroid Selection** — When multiple DBSCAN clusters appear inside the ROI (e.g. two ants briefly close together), the tracker picks the centroid **nearest to the last known position**, on the assumption that the ant cannot teleport between frames.

### ⚡ Vectorization

Multiple performance-critical operations in `Ant_Tracker.py` use NumPy vectorization instead of Python loops:

- **Color filtering** — channel comparisons (`r > 1.35 * g`, etc.) are applied as boolean array masks over the entire frame at once, replacing nested pixel-by-pixel loops.
- **Noise removal** — noise coordinates are batch-indexed into the mask array in a single operation rather than iterating point by point.
- **Fungus gap filling** — interior hole detection uses `np.logical_or.accumulate` along rows and columns, performing the full directional scan across the image in a single vectorized pass.
- **Centroid distance selection** — squared Euclidean distances from all candidate centroids to the last known position are computed as a single NumPy expression.

### Pipeline

```
Video Input
    │
    ├──► Frame 0: Color Filter + DBSCAN → All ants detected → User picks one
    │
    └──► Frame N loop:
              │
              ├──► Apply ROI mask (window around last known position)
              ├──► Color Filter + DBSCAN → Cluster(s) within window
              ├──► Proximity centroid selection → New centroid
              ├──► Shift ROI by (Δx, Δy) to follow ant
              ├──► [Optional] Check centroid against fungus filled mask
              └──► Display frame with centroid marker
                       │
                       └──► Final scatter plot of full trajectory
```

---

## 🛠️ Technologies & Libraries

| Library | Role |
|---|---|
| `opencv-python` | Frame reading, color channel access, ROI masking, bounding boxes, display |
| `scikit-learn` | `DBSCAN` used in `Ant_Tracker.py` for video-speed clustering |
| `numpy` | Vectorized color filtering, mask operations, median background, array math |
| `matplotlib` | Trajectory scatter plots, K-Means/DBSCAN cluster visualizations |

> **Note:** The notebook (`Ant_Fungus_Interaction_Analysis.ipynb`) implements K-Means and DBSCAN **entirely from scratch** using only Python and NumPy — no scikit-learn. The video tracker (`Ant_Tracker.py`) uses scikit-learn's DBSCAN for real-time performance.

---

## ⚙️ Setup

### Prerequisites
- Python **3.9 or higher**
- A video file named `ant_video.mp4` placed in the project root (for the tracker)
- Google Colab (recommended) or a local Jupyter environment (for the notebook)

### 1. Clone the repository
```bash
git clone https://github.com/LuisFelipeVelasco/Ant-Detection-and-Fungus-Analysis-Using-Machine-Learning.git
cd Ant-Detection-and-Fungus-Analysis-Using-Machine-Learning
```

### 2. Create a virtual environment
```bash
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install opencv-python scikit-learn numpy matplotlib
```

### 4. Run the video tracker
You can find the and_video.mp4 file in : https://drive.google.com/file/d/13vYUA2h4e6-osCQqmZZM_C5nuIGg-Pg6/view?usp=sharing

Place your video file in the project root as `ant_video.mp4`, then:
```bash
python Ant_Tracker.py
```
The first frame will open with all detected ants labeled. Enter the label of the ant you want to track. Press `d` at any time to stop tracking and display the final trajectory plot.

### 5. Run the image notebook
Open `Ant_Fungus_Interaction_Analysis.ipynb` in Google Colab or Jupyter. The notebook downloads its test images automatically on first run — no extra setup required.

---

## 📚 Learnings

**Clustering from Scratch (Notebook)**
- Building K-Means by hand clarifies exactly what "convergence" means: the loop terminates when the centroid update step produces no change, not after a fixed number of iterations.
- DBSCAN from scratch exposes the three-phase nature of the algorithm — core classification, border promotion, cluster expansion — which is abstracted away by library implementations.
- The `size_bias` / `b` parameter for pruning small clusters is essential: without it, minor color-noise blobs register as legitimate ants.

**Vectorization (Tracker)**
- NumPy boolean masks replace nested pixel loops entirely for color detection, cutting per-frame processing time drastically. The difference becomes visible even at moderate video resolutions.
- `np.logical_or.accumulate` is a concise way to compute "does a True exist anywhere to the left/right/above/below this cell" — one of the more elegant applications of scan operations.

**Background Subtraction**
- The median-of-frames trick for background extraction is simple yet effective: ants are in constant motion, so their pixels are statistical outliers and collapse out of the median, leaving only the static fungus region.

**ROI Tracking**
- Limiting detection to a moving window around the tracked subject reduces both computation and false positives. Shifting the window by the centroid delta each frame keeps it centered without any separate motion model.

**Proximity as a Disambiguation Heuristic**
- When two ants briefly overlap inside the ROI, picking the cluster centroid closest to the previous position is a minimal but surprisingly robust disambiguation strategy — it embeds the physical assumption that an object cannot teleport.

---

## 🚧 Known Limitations & Future Improvements

- Color-based detection is sensitive to lighting changes; an HSV-space filter or adaptive thresholding would generalize better across different environments.
- The K-Means implementation is not seeded (random initialization), so results can vary between runs — K-Means++ initialization would improve stability.
- The video tracker assumes a single ant per ROI window; simultaneous close-proximity ants can confuse the centroid selection.
- A more robust approach for video tracking could integrate a Kalman filter or lightweight deep learning detector (e.g. a fine-tuned YOLO model) to handle occlusions and fast motion.
