
# Partial-Conv-Inpainting (PConv-Inpainting)

> Image inpainting using Partial Convolution layers, based on NVIDIA's CVPR 2018 paper.

<div align="center">
  <img src="assets/example_inpainting.png" alt="Inpainting Example" width="600"/>
  <p><em>Left: Masked image | Center: Applied mask | Right: Inpainted output</em></p>
</div>

---

## 📌 Overview

**Partial-Conv-Inpainting** is a deep learning project that performs **image inpainting** using Partial Convolutional (PConv) layers. This technique intelligently fills missing regions in an image by leveraging only known pixels, making it especially powerful for irregular holes or damaged areas.

Originally proposed by **NVIDIA** in 2018, this method improves upon traditional convolutions by incorporating a binary mask that guides learning and inference.

---

## 📂 Project Structure

```
Partial-Conv-Inpainting/
├── data/              # Dataset images & masks
├── src/               # Model, training, and utils
├── checkpoints/       # Saved weights
├── results/           # Inpainting outputs
├── requirements.txt
├── train.py
├── evaluate.py
└── inference.py
```

---

## 🚀 Features

- ✅ Encoder-Decoder architecture with Partial Convolutional layers  
- ✅ Support for **irregular and free-form masks**  
- ✅ Optional GAN-based training for enhanced realism  
- ✅ Clean training/evaluation/inference pipelines  
- ✅ Easily extensible and well-documented  

---

## 🧠 How It Works

Partial Convolution only updates known pixels (based on an input binary mask), dynamically updating this mask as the image propagates through the network.

- **Encoder** compresses masked input using PartialConv2D layers.  
- **Decoder** upsamples the latent representation while continuing to refine missing areas.  
- **Mask Update**: After every PConv layer, the mask is updated to reflect which pixels have been "filled."

You may optionally use **GAN training** with:
- `Generator`: Fills in the image.
- `Discriminator`: Distinguishes real vs. fake images.

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/Partial-Conv-Inpainting.git
cd Partial-Conv-Inpainting

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

### Python Requirements

- `torch >= 1.10`
- `torchvision >= 0.11`
- `opencv-python >= 4.5`
- `Pillow`, `tqdm`, `numpy`

---

## 📁 Dataset

Use high-resolution datasets such as **CelebA-HQ**, **Places2**, or your custom dataset.

Expected folder structure:

```
data/
├── images/
│   ├── image1.jpg
│   └── ...
└── masks/
    ├── mask1.png
    └── ...
```

> No masks? No problem! Random masks can be generated automatically during training.

---

## 🔧 Training

```bash
python train.py \
  --data_root ./data/images \
  --mask_root ./data/masks \
  --batch_size 32 \
  --epochs 50 \
  --lr_g 1e-4 \
  --lr_d 1e-4 \
  --gpu_ids 0
```

- `--mask_root` is optional. If omitted, masks will be generated on-the-fly.
- Model weights will be saved under `./checkpoints/`.

---

## 📈 Evaluation

Evaluate a trained model:

```bash
python evaluate.py \
  --model_path ./checkpoints/best_model.pth \
  --data_root ./data/images \
  --mask_root ./data/masks \
  --output_dir ./results
```

---

## ✨ Inference with Pretrained Model

```bash
python inference.py \
  --image_path ./example.jpg \
  --mask_path ./example_mask.png \
  --model_path ./checkpoints/pretrained_model.pth \
  --output_path ./output.png
```

> Download pretrained models from [Releases](https://github.com/your-username/Partial-Conv-Inpainting/releases)

---

## 🧱 Model Architecture

- **Encoder**: Stack of `PartialConv2D` layers with downsampling
- **Decoder**: `Upsample + PConv` with skip connections (U-Net-like)
- Each layer updates both image features and the mask

