# Kidney Segmentation with U-Net

This module implements a U-Net-based deep learning model for segmenting kidneys in ultrasound images. The system is designed to work with the [Open Kidney Ultrasound Dataset](https://github.com/rsingla92/kidneyUS) which uses CSV-based polygon annotations from VGG Image Annotator.

## Environment Setup

1. Create a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install torch torchvision pillow opencv-python numpy pandas tqdm
```

Or install from a requirements file if available:
```bash
pip install -r requirements.txt
```

## Dataset Setup

### Downloading the Open Kidney Ultrasound Dataset

1. Register for the dataset at: https://ubc.ca1.qualtrics.com/jfe/form/SV_1TfBnLm1wwZ9srk
2. After approval, you'll receive access to the dataset
3. The dataset structure should look like:

```
kidneyUS/
  images/              # PNG ultrasound images
    image001.png
    image002.png
    ...
  labels/              # CSV annotation files from VGG Image Annotator
    annotations.csv    # Contains polygon coordinates for kidney regions
    ...
```

### Dataset Structure

The Open Kidney Ultrasound Dataset uses CSV files with VGG Image Annotator format:
- **Images**: PNG format grayscale ultrasound images
- **Annotations**: CSV files containing polygon coordinates
  - Each row represents a region annotation
  - Contains columns: `filename`, `region_shape_attributes` (with polygon coordinates)
  - Polygon coordinates are stored as JSON with `all_points_x` and `all_points_y` arrays

The dataset loader automatically:
- Parses CSV annotations to extract polygon coordinates
- Converts polygons to binary masks (0 = background, 255 = kidney)
- Handles multiple kidney regions per image
- Supports filtering by class name (e.g., 'kidney', 'native_kidney', 'transplant_kidney')

## Classical Computer Vision Approach (No Training Required)

For quick prototyping and testing without requiring model training, a classical computer vision approach is available using OpenCV.

### Quick Start

1. **Setup environment** (if not already done):
```bash
cd Backend/toshi
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install opencv-python numpy
```

2. **Run segmentation**:
```bash
python segment_kidney_cv.py \
  --image /absolute/path/to/ultrasound.png \
  --out_overlay kidney_overlay.png \
  --out_mask kidney_mask.png
```

### Algorithm Details

The algorithm pipeline:
1. **Multi-stage denoising**: Non-local means + bilateral filtering to reduce ultrasound speckle
2. **Multi-scale preprocessing**: Gaussian blur at 3 scales (fine, medium, coarse)
3. **7 detection methods**: Edge-based (Canny, gradient, morphological gradient, combined) + region-based fallbacks
4. **Smart scoring (10+ criteria)**: Size (8-30%), shape (aspect 1.5-4.5), contrast, texture, intensity mix
5. **Conservative refinement**: Region growing limited to 30%, gradient-constrained boundaries

### Arguments

- `--image`: Path to input ultrasound image (required)
- `--out_overlay`: Path to save overlay image (default: `kidney_overlay.png`)
- `--out_mask`: Path to save binary mask (default: `kidney_mask.png`)
- `--presence_thresh`: Minimum fraction of pixels for kidney presence (default: `0.01` = 1%)
- `--canny_low`: Lower threshold for Canny edge detection (default: `30`)
- `--canny_high`: Upper threshold for Canny edge detection (default: `80`)

### Output

The script prints:
- `Kidney present: True/False`
- `Mask fraction: <float>`

And saves:
- Binary mask image (0 = background, 255 = kidney)
- Overlay image with kidney region highlighted in red and yellow contour outline

### How It Works

The script uses edge-based detection optimized for kidney capsule detection:

1. **Preprocessing**: Multi-stage noise reduction (non-local means, bilateral filter) + multi-scale Gaussian blur + CLAHE contrast enhancement
2. **Edge detection**: Canny edge detection + gradient detection + morphological gradients to find kidney boundaries
3. **Contour filling**: Morphological closing to complete boundaries, then fill enclosed regions
4. **Intelligent scoring**: Each contour scored on size (8-30% optimal), shape (elongated, 1.5-4.5 aspect ratio), location, contrast, and intensity pattern (mixed bright + dark)
5. **Mask refinement**: Conservative region growing with gradient constraints + boundary smoothing

**Key features**:
- Detects complete kidney (bright capsule + dark interior)
- Handles both hyperechoic and hypoechoic kidneys
- Works on diagonal/elongated kidney views
- Edge-based approach finds accurate boundaries

### Note

**This is a working classical CV implementation** that requires no training data. It accurately detects kidney structures in ultrasound images by finding the kidney capsule boundary. For specialized clinical use cases, it can be replaced with a trained CNN/U-Net model.

## Training the Model

Train the U-Net model on the Open Kidney Ultrasound Dataset:

```bash
python train_kidney_unet.py \
  --images_dir kidneyUS/images \
  --labels_csv kidneyUS/labels/annotations.csv \
  --epochs 50 \
  --batch_size 8 \
  --lr 1e-3 \
  --img_size 256 \
  --val_split 0.2 \
  --out kidney_unet.pth
```

### Training Arguments

- `--images_dir`: Directory containing ultrasound images (required)
- `--labels_csv`: Path to CSV file with VGG Image Annotator annotations (required)
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 8)
- `--lr`: Learning rate (default: 1e-3)
- `--img_size`: Image size for resizing (default: 256)
- `--val_split`: Validation split ratio (default: 0.2)
- `--num_workers`: Number of data loading workers (default: 4)
- `--class_name`: Class name to extract from annotations (default: 'kidney')
- `--out`: Path to save model checkpoint (default: kidney_unet.pth)

The model will be saved automatically when validation Dice score improves.

**Note**: The dataset loader converts polygon annotations to binary masks on-the-fly during training. This means you don't need to pre-process the CSV files into mask images.

## Running Inference

### Interactive Mode (Recommended)

The easiest way to run inference is using the interactive script that prompts you for image upload:

```bash
python run_inference.py
```

This will:
- Prompt you to enter or drag-and-drop an image path
- Ask for the model path (or use default `kidney_unet.pth`)
- Run inference and display results
- Save mask and overlay images automatically
- Allow you to analyze multiple images in one session

**No dataset required** - just upload your ultrasound image and get instant results!

### Command-Line Mode

Alternatively, run inference with command-line arguments:

```bash
python infer_kidney_unet.py \
  --image path/to/ultrasound_image.png \
  --model kidney_unet.pth \
  --out_mask kidney_mask.png \
  --out_overlay kidney_overlay.png \
  --img_size 256 \
  --area_threshold 0.01
```

### Inference Arguments

- `--image`: Path to input ultrasound image (required)
- `--model`: Path to trained model checkpoint (required)
- `--out_mask`: Path to save binary mask (default: kidney_mask.png)
- `--out_overlay`: Path to save overlay image (default: kidney_overlay.png)
- `--img_size`: Image size for model input (default: 256)
- `--area_threshold`: Minimum fraction of pixels for kidney presence (default: 0.01 = 1%)

### Kidney Presence Detection

The inference script determines if a kidney is present based on the mask area:

- **Fraction calculation**: `kidney_pixels / total_pixels`
- **Presence threshold**: If the fraction ≥ `area_threshold`, kidney is considered present
- **Default threshold**: 0.01 (1% of image pixels must be kidney)

The script outputs:
- `Kidney present: True/False`
- `Mask fraction: <value>`
- Binary mask image (0/255)
- Overlay image with kidney region highlighted in red

## Model Architecture

The U-Net architecture consists of:

- **Encoder**: Downsampling path with DoubleConv blocks and MaxPool
- **Bottleneck**: DoubleConv at the bottom
- **Decoder**: Upsampling path with ConvTranspose2d and skip connections
- **Output**: Single-channel logits (sigmoid applied during inference)

## Loss Function

The model uses a combined loss function:
- **BCE Loss**: Binary Cross-Entropy with logits for stable gradients
- **Dice Loss**: Focuses on overlap between prediction and target
- **Combined**: `BCE + Dice` (equal weights)

## Dataset Format Details

### VGG Image Annotator CSV Format

The CSV files from VGG Image Annotator typically contain:
- `filename`: Image filename
- `file_size`: Size of the image file
- `file_attributes`: JSON with file-level attributes
- `region_count`: Number of regions in the image
- `region_id`: Region identifier
- `region_shape_attributes`: JSON with polygon coordinates:
  ```json
  {
    "name": "polygon",
    "all_points_x": [x1, x2, x3, ...],
    "all_points_y": [y1, y2, y3, ...]
  }
  ```
- `region_attributes`: JSON with region class labels

The dataset loader automatically handles variations in column names and formats.

## Files

- `segment_kidney_cv.py`: **Classical CV segmentation script** (no training required)
- `unet_kidney.py`: U-Net model implementation
- `dataset_kidney.py`: Dataset loader with CSV polygon parsing
- `losses_metrics.py`: Loss functions and metrics
- `train_kidney_unet.py`: Training script
- `infer_kidney_unet.py`: Command-line inference script
- `run_inference.py`: **Interactive inference script** (prompts for image upload)

## References

- **Open Kidney Ultrasound Dataset**: [GitHub Repository](https://github.com/rsingla92/kidneyUS)
- **Citation**: Singla, R., Ringstrom, C., Hu, G., Lessoway, V., Reid, J., Nguan, C., & Rohling, R. (2023, October). The open kidney ultrasound data set. In International Workshop on Advances in Simplifying Medical Ultrasound (pp. 155-164). Cham: Springer Nature Switzerland.
