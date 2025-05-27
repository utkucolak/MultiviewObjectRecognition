# Multiview Object Recognition

This project implements multiview object recognition using handcrafted feature detection and description methods inspired by SIFT and SURF. It matches a query image to one of many views across multiple object folders (e.g., COIL-100 format) using Harris keypoints, histogram-based descriptors, and a ratio-test matcher — all without OpenCV modules.

---

## ✨ Features

- 🔍 **Custom Keypoint Detection** using Harris corner response
- 📐 **SIFT-like Descriptor** using orientation histograms over local patches
- 🤝 **Ratio-based Feature Matching** for robust correspondence
- ⚡ **Parallel Processing** for scalable multiview comparisons
- 📷 **COIL-100-style Dataset** support

---

## 📁 Folder Structure

```
MultiviewObjectRecognition/
├── coil-100/              # (ignored) Full COIL-100 dataset
├── dataset/               # (ignored) Subset used for testing
├── features/              # Detector, Descriptor, Matcher modules
├── utils/                 # I/O utilities for visualizing matches
├── main.py                # Main matching pipeline
├── matches_result.png     # Last matching result saved image
├── requirements.txt       # Python dependencies
└── readme.md              # Project README
```

> ⚠️ Note: `dataset/` and `coil-100/` are ignored via `.gitignore`.

---

## 🛠️ Installation

```bash
git clone https://github.com/yourusername/MultiviewObjectRecognition.git
cd MultiviewObjectRecognition
pip install -r requirements.txt
```

> Requires Python 3.8+ and OpenCV + NumPy + tqdm

---

## 🚀 Usage

By default, `main.py` matches `dataset/obj_20/60.png` against all views of the first 20 objects:

```bash
python main.py
```

- Outputs `[DONE]` logs and `[RESULT] Best match` summary
- Saves the visualization as `matches_result.png`

You can change:
- `QUERY_IMG_PATH` in `main.py` to test a different query
- `MAX_OBJECTS` or `MAX_VIEWS_PER_OBJECT` to scale up

---

## 📊 Output Example

If match is successful, `matches_result.png` will show feature correspondences between query and best-matching view.

---

## ✅ How it Works

1. Detects Harris corners in query and dataset images
2. Computes SIFT-like descriptors over \( 8 	imes 8 \) patches (4×4 cells, 8-bin histograms)
3. Matches descriptors using Lowe's ratio test
4. Picks the view with the highest number of good matches

---

## 🤝 Contributing

If you'd like to help improve the project, feel free to fork and PR! Suggestions welcome for:
- Rotation-invariant descriptors
- Affine normalization
- ANN-based matchers

---

## 📬 Contact

Created by **Mehmet Utku Çolak**  
📧 colakme19@itu.edu.tr
