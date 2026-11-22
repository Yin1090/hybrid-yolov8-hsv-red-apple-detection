# Hybrid YOLOv8n + Adaptive HSV Red Apple Detection

[English](#english) | [Tiếng Việt](#tiếng-việt)

---

## English

### 1. Overview

**Hybrid YOLOv8n + Adaptive Split-Range HSV Red Apple Detection** is a 4-layer coarse-to-fine computer vision pipeline designed to detect and count red apples (`red_apple`) reliably under orchard foliage occlusion, dense crate clutter with non-apple distractors, and translucent retail packaging with specular glare.

Pure deep-learning detectors (`YOLO`) can propose non-red objects (orange citrus, crate stickers, ribbons) under relaxed confidence thresholds, while pure color segmentation (`HSV`) struggles with illumination shifts and specular highlights. This hybrid architecture combines global object proposals with local illumination-normalized HSV segmentation, morphological glare-bridging, geometric admissibility gates, weighted score fusion, and secondary Non-Maximum Suppression (NMS).

Key capabilities:
- **Pluggable Layer 2 Backends**: Supports `UltralyticsBackend` (`.pt`), `ONNXRuntimeBackend` (`.onnx`), and an offline pixel-driven `CalibratedProposerBackend` (OpenCV + NumPy only, zero external model weights required).
- **4 Operating Modes**: `hybrid` (`A:HYBRID`), `yolo` (`Y:YOLO`), `hsv` (`C:HSV`), and `auto` (`H:AUTO`).
- **3-Panel Visualization**: Side-by-side `Original | Mask (HSV/ROI) | Final Result` renderer with interactive `ROI_HALF = 100` HSV calibration (`--show`) and headless PNG/JSON export.

---

### 2. Architecture & Datasets

#### 2.1 Four-Layer Coarse-to-Fine Pipeline

1. **Layer 1 — Acquisition & Normalization (`acquisition.py`)**
   - Loads static images (with Unicode-safe `np.fromfile` + `cv2.imdecode` fallback), sorted image directories, video files, and webcam streams (`cv2.VideoCapture`).
   - Normalizes inputs to contiguous 3-channel `uint8` BGR (`normalize_bgr`).

2. **Layer 2 — Global Perception (`perception.py`)**
   - Proposes candidate bounding boxes `(x1, y1, x2, y2)` with confidence `yconf`.
   - Configurable confidence thresholds: `--yolo-conf` (`0.15` high-recall candidate mode, `0.65` standard threshold, `0.85` strict ablation) and `--bypass-conf` (`0.65` default).
   - Offline `CalibratedProposerBackend` uses warm-red chromatic prominence (`red_prom = 0.80*(R - G) + 0.20*(R - B)`), distance-transform peak splitting, and local patch edge/glare scoring without relying on any external weights or ground-truth files.

3. **Layer 3 — Local ROI Refinement & Adaptive HSV (`refinement.py`)**
   - Per-ROI preprocessing: `GaussianBlur((5, 5), 0)` → `BGR2HSV` → **CLAHE** (`clipLimit=2.0`, `tileGridSize=(8, 8)`) applied strictly to the **`V` channel**.
   - **Adaptive Split-Range Red Hue**: `[0, 15]` and `[160, 179]` baseline; interactive/programmatic ROI calibration (`ROI_HALF = 100`, `200×200` patch) adapts `S`/`V` via `20th`/`80th` percentiles and wraps `H` bounds (`[ROI CLICK]`, `[HSV CALC]`).
   - **Morphology**: `medianBlur(mask, 5)` → `MORPH_OPEN` (`7×7` `cv2.MORPH_ELLIPSE`, `iterations=1`) → `MORPH_CLOSE` (`7×7` `cv2.MORPH_ELLIPSE`, `iterations=3`) to bridge specular glare holes.
   - **Physical & Geometric Admissibility Gate** (executed before Layer 4 bypass):
     - `red_ratio > 0.08` (`> 8%` foreground pixels)
     - `Circularity > 0.25` (`4π·Area / Perimeter²`)
     - `Solidity > 0.60` (`Area / ConvexHullArea`)
     - `Extent > 0.40` (`Area / BoundingBoxArea`)
     - `Aspect Ratio ∈ [0.5, 2.0]` (`width / height`)
     - Candidates failing any check are rejected immediately.

4. **Layer 4 — Hybrid Fusion & Secondary NMS (`fusion.py`)**
   - Fusion rule for physically admissible candidates:
     ```python
     if yconf > 0.65:
         Score_final = yconf
     else:
         Score_final = float(np.clip(0.45 * yconf + 0.55 * hsv_conf, 0.0, 1.0))
     ```
   - Acceptance threshold: `Score_final >= 0.35`.
   - Secondary NMS (`nms_hybrid_boxes`): sorts descending by `Score_final` and suppresses boxes with `IoU > 0.55`.

#### 2.2 Datasets & Calibrated Benchmarks (`data/`)

- **`data/results.csv`** (100-epoch YOLOv8n `red_apple` training log, 14 columns):
  - Peak `Precision`: **`0.9442`** @ epoch `55`
  - Peak `Recall`: **`0.9103`** @ epoch `16`
  - Peak `mAP@0.5`: **`0.9512`** @ epoch `65`
  - Peak `mAP@0.5:0.95`: **`0.5605`** @ epoch `81`
- **`data/pr_f1_summary.json`**:
  - Peak `F1 = 0.88` @ confidence `0.491` (plateau `0.82–0.88` across `[0.1, 0.8]`)
  - `AP@0.5 = 0.940`, `Recall@0.0 = 0.96`
- **Synthetic Test Scenes (`data/samples/`)**:

| Scene | GT Apples | `yolo_conf=0.65` (YOLO) | `yolo_conf=0.65` (Hybrid) | `yolo_conf=0.85` (Hybrid) | `close_iter=0` Ablation |
|---|---:|---:|---:|---:|---:|
| `orchard_scene.png` | 12 | 12 | 12 (`F1=1.000`) | 12 | — |
| `dense_crate_scene.png` | 46 (+10 distractors) | 56 (`P=0.821`) | 46 (`P=1.000, R=1.000`) | 20 (`R=0.435`) | — |
| `bagged_apples_scene.png` | 16 (glare + veil) | 16 | 16 (`F1=1.000`) | 4 (`R=0.250`) | 4 (`R=0.250`) |
| `negative_foliage_scene.png` | 0 | 0 | 0 | 0 | — |

---

### 3. Installation & Usage

#### Prerequisites & Setup
```bash
python -m pip install -r requirements.txt
# Optional: install package in editable mode + YOLOv8/ONNX extras
python -m pip install -e ".[dev]"
# Optional inference runtimes:
# python -m pip install -e ".[onnx]"
# python -m pip install -e ".[yolo]"
```

#### CLI Usage (Headless & Interactive)
> Note: If you skipped `python -m pip install -e ".[dev]"`, set `PYTHONPATH=src` before invoking `python -m hybrid_apple_detector`.

```bash
# Run headless hybrid detection on a single image and save 3-panel output + JSON
python -m hybrid_apple_detector --source data/samples/dense_crate_scene.png --mode hybrid --yolo-conf 0.65 --output-dir outputs --save-json

# Run batch detection across all sample scenes with ROI_HALF=100 HSV calibration click
python -m hybrid_apple_detector --source data/samples --roi-click 190,155 --output-dir outputs --save-json

# Launch interactive OpenCV 3-panel GUI window (requires desktop display)
python -m hybrid_apple_detector --source data/samples/bagged_apples_scene.png --show
```

Interactive window title & hotkeys:
`HYBRID VIP APPLE | Y:YOLO | H:AUTO | C:HSV | A:HYBRID | S:SAVE | R:RESET`

#### Regenerate Datasets & Run Analysis / Ablation Benchmarks
```bash
python scripts/generate_dataset.py
python scripts/analyze_results.py
python scripts/run_benchmark.py
```

---

### 4. Automated Tests

Run the full offline, headless `pytest` verification suite from the repository root:
```bash
python -m pytest -q
```
The suite validates Layer 1 acquisition (images, Unicode paths, directories, synthetic `.avi` video), Layer 2 backends (mocked ONNX/Ultralytics decode + filename/flip/pad invariance proving zero hardcoding), Layer 3 CLAHE-V/split-range HSV/morphology/geometry filters, Layer 4 fusion & NMS (`IoU=0.55`), `results.csv` training peaks, 3-panel visualization, and end-to-end CLI/benchmark scenarios.

---

## Tiếng Việt

### 1. Tổng quan

**Hybrid YOLOv8n + Adaptive Split-Range HSV Red Apple Detection** là pipeline thị giác máy tính 4 tầng từ thô đến tinh (coarse-to-fine) giúp phát hiện và đếm táo đỏ (`red_apple`) ổn định trong điều kiện tán lá che khuất tại vườn, khay/thùng mật độ cao có vật gây nhiễu, và táo bọc túi nilon kèm dải lóa sáng (specular glare).

Bộ dò thuần học sâu (`YOLO`) dễ nhận nhầm các vật thể không phải táo (cam vàng, tem nhãn, ruy-băng đỏ) khi hạ ngưỡng tin cậy, trong khi tách màu thuần túy (`HSV`) lại nhạy cảm với thay đổi ánh sáng và vùng chói sáng. Kiến trúc lai kết hợp đề xuất vùng toàn cục với chuẩn hóa ánh sáng cục bộ, tách ngưỡng HSV hai dải, xử lý hình thái học lấp lỗ chói, bộ lọc hình học vật lý, hợp nhất điểm số có trọng số và NMS thứ cấp.

Tính năng chính:
- **Backend Tầng 2 linh hoạt**: Hỗ trợ `UltralyticsBackend` (`.pt`), `ONNXRuntimeBackend` (`.onnx`), và `CalibratedProposerBackend` chạy hoàn toàn offline bằng OpenCV + NumPy (không phụ thuộc trọng số hay dữ liệu nhãn).
- **4 chế độ hoạt động**: `hybrid` (`A:HYBRID`), `yolo` (`Y:YOLO`), `hsv` (`C:HSV`), và `auto` (`H:AUTO`).
- **Trực quan hóa 3 khung hình**: Hiển thị song song `Original | Mask (HSV/ROI) | Final Result`, hỗ trợ hiệu chỉnh HSV tương tác theo vùng `ROI_HALF = 100` (`--show`) và xuất ảnh/JSON không cần màn hình (headless).

---

### 2. Kiến trúc & Dữ liệu

#### 2.1 Kiến trúc Pipeline 4 Tầng

1. **Tầng 1 — Thu nhận & Chuẩn hóa (`acquisition.py`)**
   - Đọc ảnh đơn (hỗ trợ đường dẫn Unicode qua `np.fromfile` + `cv2.imdecode`), thư mục ảnh theo thứ tự, tệp video và webcam (`cv2.VideoCapture`).
   - Chuẩn hóa về ảnh BGR `uint8` 3 kênh liên tục (`normalize_bgr`).

2. **Tầng 2 — Nhận thức Toàn cục (`perception.py`)**
   - Đề xuất các khung bao ứng viên `(x1, y1, x2, y2)` kèm độ tin cậy `yconf`.
   - Ngưỡng cấu hình độc lập: `--yolo-conf` (`0.15` chế độ quét bao phủ cao, `0.65` ngưỡng chuẩn, `0.85` thử nghiệm cắt giảm) và `--bypass-conf` (`0.65`).
   - `CalibratedProposerBackend` tính độ nổi bật sắc đỏ (`red_prom = 0.80*(R - G) + 0.20*(R - B)`), tách cụm bằng Distance Transform và chấm điểm biên/độ chói cục bộ hoàn toàn từ dữ liệu điểm ảnh.

3. **Tầng 3 — Tinh chỉnh ROI Cục bộ & HSV Thích ứng (`refinement.py`)**
   - Tiền xử lý mỗi ROI: `GaussianBlur((5, 5), 0)` → `BGR2HSV` → **CLAHE** (`clipLimit=2.0`, `tileGridSize=(8, 8)`) áp dụng **duy nhất trên kênh `V`**.
   - **Tách ngưỡng Hue đỏ hai dải (Split-Range)**: `[0, 15]` và `[160, 179]`; hiệu chỉnh ROI (`ROI_HALF = 100`, vùng `200×200`) thích ứng `S`/`V` theo bách phân vị thứ `20` và `80` (`[ROI CLICK]`, `[HSV CALC]`).
   - **Hình thái học**: `medianBlur(mask, 5)` → `MORPH_OPEN` (`7×7` Ellipse, `iterations=1`) → `MORPH_CLOSE` (`7×7` Ellipse, `iterations=3`) để nối liền vùng bị lóa sáng.
   - **Bộ lọc hợp lệ vật lý & hình học** (thực thi trước nhánh bypass của Tầng 4):
     - `red_ratio > 0.08` (`> 8%` diện tích điểm ảnh đỏ)
     - `Circularity > 0.25` (`4π·Area / Perimeter²`)
     - `Solidity > 0.60` (`Area / ConvexHullArea`)
     - `Extent > 0.40` (`Area / BoundingBoxArea`)
     - `Aspect Ratio ∈ [0.5, 2.0]` (`width / height`)
     - Loại bỏ ngay lập tức các ứng viên không đạt ngưỡng.

4. **Tầng 4 — Hợp nhất Quyết định & NMS Thứ cấp (`fusion.py`)**
   - Công thức hợp nhất cho ứng viên đạt Tầng 3:
     ```python
     if yconf > 0.65:
         Score_final = yconf
     else:
         Score_final = float(np.clip(0.45 * yconf + 0.55 * hsv_conf, 0.0, 1.0))
     ```
   - Ngưỡng giữ lại: `Score_final >= 0.35`.
   - NMS thứ cấp (`nms_hybrid_boxes`): sắp xếp giảm dần theo `Score_final` và loại bỏ khung chồng lấn có `IoU > 0.55`.

#### 2.2 Bộ dữ liệu & Kết quả Kiểm chuẩn (`data/`)

- **`data/results.csv`** (nhật ký huấn luyện YOLOv8n lớp `red_apple` 100 epoch, 14 cột):
  - `Precision` cực đại: **`0.9442`** tại epoch `55`
  - `Recall` cực đại: **`0.9103`** tại epoch `16`
  - `mAP@0.5` cực đại: **`0.9512`** tại epoch `65`
  - `mAP@0.5:0.95` cực đại: **`0.5605`** tại epoch `81`
- **`data/pr_f1_summary.json`**:
  - `F1 = 0.88` tại ngưỡng tin cậy `0.491` (vùng bình nguyên `0.82–0.88` trên `[0.1, 0.8]`)
  - `AP@0.5 = 0.940`, `Recall@0.0 = 0.96`
- **Bộ ảnh kiểm thử tổng hợp (`data/samples/`)**:

| Kịch bản | Số táo thực tế (GT) | `yolo_conf=0.65` (YOLO) | `yolo_conf=0.65` (Hybrid) | `yolo_conf=0.85` (Hybrid) | `close_iter=0` (Không đóng) |
|---|---:|---:|---:|---:|---:|
| `orchard_scene.png` | 12 | 12 | 12 (`F1=1.000`) | 12 | — |
| `dense_crate_scene.png` | 46 (+10 nhiễu) | 56 (`P=0.821`) | 46 (`P=1.000, R=1.000`) | 20 (`R=0.435`) | — |
| `bagged_apples_scene.png` | 16 (túi + lóa) | 16 | 16 (`F1=1.000`) | 4 (`R=0.250`) | 4 (`R=0.250`) |
| `negative_foliage_scene.png` | 0 | 0 | 0 | 0 | — |

---

### 3. Cài đặt & Sử dụng

#### Cài đặt môi trường
```bash
python -m pip install -r requirements.txt
# Tùy chọn cài đặt ở chế độ editable kèm công cụ kiểm thử:
python -m pip install -e ".[dev]"
# Tùy chọn runtime suy luận:
# python -m pip install -e ".[onnx]"
# python -m pip install -e ".[yolo]"
```

#### Chạy qua dòng lệnh (Headless & Tương tác)
> Lưu ý: Nếu chưa cài gói bằng `python -m pip install -e ".[dev]"`, hãy đặt `PYTHONPATH=src` trước khi chạy `python -m hybrid_apple_detector`.

```bash
# Chạy phát hiện Hybrid không giao diện trên một ảnh và lưu kết quả 3 khung + JSON
python -m hybrid_apple_detector --source data/samples/dense_crate_scene.png --mode hybrid --yolo-conf 0.65 --output-dir outputs --save-json

# Chạy hàng loạt trên thư mục mẫu kèm hiệu chỉnh điểm ROI_HALF=100
python -m hybrid_apple_detector --source data/samples --roi-click 190,155 --output-dir outputs --save-json

# Mở cửa sổ OpenCV tương tác 3 khung hình (yêu cầu màn hình đồ họa)
python -m hybrid_apple_detector --source data/samples/bagged_apples_scene.png --show
```

Tiêu đề cửa sổ & phím tắt tương tác:
`HYBRID VIP APPLE | Y:YOLO | H:AUTO | C:HSV | A:HYBRID | S:SAVE | R:RESET`

#### Tái tạo dữ liệu & Chạy phân tích / Benchmark
```bash
python scripts/generate_dataset.py
python scripts/analyze_results.py
python scripts/run_benchmark.py
```

---

### 4. Kiểm thử tự động

Chạy toàn bộ bộ kiểm thử `pytest` hoàn toàn offline từ thư mục gốc repository:
```bash
python -m pytest -q
```
Bộ kiểm thử xác minh đầy đủ Tầng 1 (đọc ảnh, đường dẫn Unicode, thư mục, video `.avi`), Tầng 2 (giải mã ONNX/Ultralytics giả lập và kiểm tra bất biến đổi tên/lật ảnh/đệm viền chống gian lận), Tầng 3 (CLAHE-V, HSV hai dải, hình thái học lấp lóa sáng, 4 bộ lọc hình học), Tầng 4 (công thức hợp nhất và `nms_hybrid_boxes` tại `IoU=0.55`), các đỉnh chỉ số `results.csv`, xuất ảnh 3 khung và luồng CLI đầu-cuối.
