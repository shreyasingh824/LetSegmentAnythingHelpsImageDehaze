
# Let Segment Any thing helps Image Dehaze

This project involves image restoration using a combination of convolutional neural networks (CNNs) and Facebook Research's Segment Anything Model. The goal is to enhance images degraded by haze or other atmospheric conditions. 

## The project includes two main sections:
### 1.Image Restoration without SAM (Segment Anything Model):
1. Utilizes a CNN-based encoder-decoder architecture to restore hazy images.

2. Training involves optimizing Mean Squared Error (MSE) loss between original and restored images.

### 2.Image Restoration with SAM:

1. Incorporates Facebook Research's Segment Anything Model (SAM) for image segmentation.

2. SAM helps in better understanding image context, potentially enhancing restoration quality.
Similar to the first section, utilizes an encoder-decoder architecture for training.


## Libraries and Frameworks Used

### Python Libraries:
   1.  OpenCV (cv2)
   2. NumPy
   3. tqdm
   4. matplotlib
   5. PyTorch


### Frameworks:
   1. Facebook Research's Segment Anything Model (SAM).

## Installation

To run this project, make sure you have the required libraries installed. You can install them using pip:

```bash
!pip install git+https://github.com/facebookresearch/segment-anything.git
!pip install opencv-python pycocotools matplotlib onnxruntime onnx
!{sys.executable} -m pip install opencv-python matplotlib
!{sys.executable} -m pip install 'git+https://github.com/facebookresearch/segment-anything.git'

```
    
Additionally, ensure you have the necessary pre-trained model checkpoints. You can download them using wget:

```

!wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth


```
## Usage

### Preparation:
Organize your dataset of hazy and ground truth images. Adjust file paths accordingly in the script.
### Training:
Execute the script provided, adjusting hyperparameters as needed.
### Results:
The script will generate loss plots for both methods (with and without SAM) for comparison.


## Note

1. Ensure GPU availability for faster training, as the script utilizes CUDA for GPU acceleration.
2. For optimal performance, consider experimenting with different architectures, hyperparameters, and training strategies.
