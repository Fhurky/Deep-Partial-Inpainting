import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import vgg19, VGG19_Weights
from PIL import Image
import os
import numpy as np
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import random # Maske oluşturma için
import cv2 # Serbest biçimli maske oluşturma için (PIL'den daha kolay)

# --- 1. Global Sabitler ve Yapılandırma Parametreleri ---

# Eğitim parametreleri
BATCH_SIZE = 16 # GPU belleğinize göre ayarlayın, 512x512 için daha düşük batch size gerekebilir
NUM_WORKERS = 4 # Veri yüklemede kullanılacak çekirdek sayısı
EPOCHS = 8 # Toplam eğitim epoch sayısı

LEARNING_RATE_G = 1e-4
LEARNING_RATE_D = 1e-4

# Çözünürlük
IMG_SIZE = (512, 512) # Giriş ve Çıkış Görüntü Boyutu

# Veri klasörünüzün yolu
BASE_DATASET_DIR = "./data" # Buraya kendi veri setinizin yolunu yazın
# Örneğin: BASE_DATASET_DIR = "./CelebA-HQ/train"

# Kayıp Fonksiyonu Ağırlıkları
LAMBDA_PIXEL = 1.0 # Piksel (L1) kaybına verilen ağırlık (inpainting için genelde yüksektir)
LAMBDA_CONTENT = 1.0 # Algısal (VGG) kaybına verilen ağırlık
LAMBDA_ADV = 0.01 # Jeneratörün adverseryal kaybına verilen ağırlık (RaGAN için ayarlanabilir)

# Cihaz (GPU varsa GPU, yoksa CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan Cihaz: {DEVICE}")

# Çıktı klasörünü oluştur
os.makedirs("./inpainting_results", exist_ok=True)
os.makedirs("./inpainting_results/samples", exist_ok=True) # Örnek çıktıları kaydetmek için

# --- 2. Yardımcı Fonksiyonlar ---

def normalize_img_to_neg1_1(img):
    """Görüntüyü [0, 255]'ten [-1, 1] aralığına normalize eder."""
    return (img / 127.5) - 1.0

def denormalize_img_from_neg1_1(img):
    """Görüntüyü [-1, 1]'den [0, 255] aralığına denormalize eder."""
    return (img + 1) * 127.5

def create_random_mask(img_size=(512, 512), mask_type="irregular", max_masks=3, max_size_ratio=0.3, max_thickness=20):
    """
    Belirtilen boyutta rastgele bir maske oluşturur.
    mask_type: "rectangle" (dikdörtgen) veya "irregular" (serbest biçimli)
    """
    mask = np.zeros(img_size, dtype=np.uint8)

    if mask_type == "rectangle":
        num_masks = random.randint(1, max_masks)
        for _ in range(num_masks):
            w, h = img_size
            max_mask_w = int(w * max_size_ratio)
            max_mask_h = int(h * max_size_ratio)

            mask_w = random.randint(int(w * 0.05), max_mask_w)
            mask_h = random.randint(int(h * 0.05), max_mask_h)

            x = random.randint(0, w - mask_w)
            y = random.randint(0, h - mask_h)
            mask[y:y+mask_h, x:x+mask_w] = 255 # Beyaz (maske)
    
    elif mask_type == "irregular":
        # Daha gerçekçi serbest biçimli maskeler için
        # Bu kısım daha karmaşık bir maske üretici gerektirir.
        # Basit bir çizim algoritması veya hazır kütüphaneler kullanılabilir.
        # Burada basit bir fırça darbesi simülasyonu yapalım.
        num_strokes = random.randint(5, 15)
        for _ in range(num_strokes):
            x1 = random.randint(0, img_size[0])
            y1 = random.randint(0, img_size[1])
            x2 = random.randint(0, img_size[0])
            y2 = random.randint(0, img_size[1])
            thickness = random.randint(5, max_thickness)
            
            # cv2.line, cv2.circle gibi fonksiyonları kullanabiliriz
            # PIL ile çizim yapmak isterseniz ImageDraw kullanabilirsiniz
            # Ancak numpy maskeyi PIL'e çevirip çizim yapıp geri numpy'a dönüştürmek gerekir.
            # Şimdilik numpy üzerinde basit bir çizim yapalım (bu kısım daha geliştirilebilir)
            cv2.line(mask, (x1, y1), (x2, y2), 255, thickness)
            cv2.circle(mask, (x1, y1), thickness // 2, 255, -1) # Yuvarlak başlangıç/bitişler

        # Maskenin boşluklarını doldurma veya büyütme (isteğe bağlı)
        mask = cv2.dilate(mask, np.ones((random.randint(1,5), random.randint(1,5)), np.uint8), iterations=random.randint(1, 3))
        mask = cv2.erode(mask, np.ones((random.randint(1,5), random.randint(1,5)), np.uint8), iterations=random.randint(1, 3))
        mask = cv2.blur(mask, (random.randint(1,5), random.randint(1,5))) # Kenarları yumuşatma

    return mask.astype(bool) # Boolean maskeye çevir

class InpaintingDataset(Dataset):
    def __init__(self, data_dir, img_size=(512, 512), mask_type="irregular", max_masks=3, max_size_ratio=0.3, max_thickness=20):
        self.data_dir = data_dir
        self.img_size = img_size
        self.mask_type = mask_type
        self.max_masks = max_masks
        self.max_size_ratio = max_size_ratio
        self.max_thickness = max_thickness

        image_extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff')
        self.image_paths = []
        for ext in image_extensions:
            self.image_paths.extend(glob.glob(os.path.join(self.data_dir, ext)))
        self.image_paths.sort()

        if not self.image_paths:
            print(f"Uyarı: '{data_dir}' dizininde işlenecek görüntü bulunamadı. Lütfen dizin yolunu ve dosya uzantılarını kontrol edin.")
            print("Örnek: CelebA-HQ gibi bir veri setini indirdiğinizden ve doğru dizine yerleştirdiğinizden emin olun.")


        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(), # [0, 1] aralığına normalize eder
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        original_img = Image.open(img_path).convert("RGB")
        original_img = self.transform(original_img) # [0, 1] aralığında Tensor

        # Maske oluştur
        # create_random_mask fonksiyonu numpy boolean maske döndürür
        mask_np = create_random_mask(self.img_size, self.mask_type, self.max_masks, self.max_size_ratio, self.max_thickness)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).float() # [0, 1] aralığında (1, H, W)

        # Maskeyi görüntünün üzerine uygula (maskelenmiş bölgeyi siyaha çevir)
        # Giriş resmi ve maske tensor'larının boyutları aynı olmalı
        # mask_tensor'ı (C, H, W) yapmak için genişlet
        mask_tensor_expanded = mask_tensor.expand_as(original_img) # (3, H, W)
        masked_img = original_img * (1 - mask_tensor_expanded) # Maskeli bölgeler 0 olur

        # Görüntüleri [-1, 1] aralığına normalize et
        original_img_norm = normalize_img_to_neg1_1(original_img * 255.0)
        masked_img_norm = normalize_img_to_neg1_1(masked_img * 255.0)

        # Maskeyi de tensöre çevir ve normalleştir (0 veya 1 olarak kalsın, model için maske bilgisi)
        # Kısmi evrişim için maske tensor'unun değerleri 0 (maskeli) veya 1 (maskesiz) olmalıdır.
        mask_tensor_final = mask_tensor.bool().float() # Maske olarak 0.0 ve 1.0

        return masked_img_norm, original_img_norm, mask_tensor_final

def create_dataloader(data_dir, batch_size, num_workers, img_size):
    dataset = InpaintingDataset(data_dir, img_size)
    if not dataset.image_paths:
        return None

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader

# --- 3. Model Mimarileri (Üreteç ve Ayırt Edici) ---

# Kısmi Evrişim Katmanı
class PartialConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(PartialConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                            padding, dilation, groups, bias)
        self.mask_kernel = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1], device=DEVICE)
        self.slide_winsize = self.kernel_size[0] * self.kernel_size[1]
        
    def forward(self, input, mask):
        # input: (B, C_in, H_in, W_in)
        # mask: (B, C_in, H_in, W_in)

        single_channel_mask_for_conv = mask[:, 0:1, :, :] # (B, 1, H_in, W_in)
        
        with torch.no_grad():
            # update_mask will have shape (B, 1, H_out, W_out)
            update_mask = F.conv2d(single_channel_mask_for_conv, self.mask_kernel, bias=None, stride=self.stride,
                                   padding=self.padding, dilation=self.dilation, groups=1)
        
        mask_ratio = self.slide_winsize / (update_mask + 1e-8) # (B, 1, H_out, W_out)
        
        # Clamp update_mask for the next layer's mask propagation
        # This update_mask will be (B, 1, H_out, W_out)
        update_mask_for_next_layer = torch.clamp(update_mask, 0, 1)

        # Apply convolution to the input features, masked.
        # This `output_features` will have shape (B, C_out, H_out, W_out)
        output_features = super(PartialConv2d, self).forward(input * mask)
        
        # Expand mask_ratio to match the output_features.
        # This is the crucial part. mask_ratio's spatial dimensions (H_out, W_out)
        # must already match output_features' spatial dimensions.
        # If they do, expand_as will only expand the channel dimension.
        # If they don't, it means a spatial mismatch occurred somewhere earlier (e.g., in update_mask calculation).
        
        # The previous suggestion was `mask_ratio.expand_as(output)`.
        # Your original code was `mask_ratio.expand_as(input[:, :self.out_channels, :, :])`.
        # The problem is that `input` has `H_in, W_in` dimensions, while `mask_ratio` and `output_features` have `H_out, W_out`.
        # So you need to expand `mask_ratio` to the spatial dimensions of the *output* features and the channel dimensions of the *output* features.
        
        # The correct target for `expand_as` for `mask_ratio` is `output_features`.
        # This assumes that `update_mask` (and thus `mask_ratio`) has the same spatial dimensions as `output_features`.
        # They should if the stride/padding are applied consistently.
        output_features = output_features * mask_ratio.expand_as(output_features)
        
        # The output mask for the next layer should have self.out_channels.
        # We expand the single-channel `update_mask_for_next_layer` to `out_channels`.
        return output_features, update_mask_for_next_layer.expand(-1, self.out_channels, -1, -1)

# PConvUNet (Partial Convolution U-Net) Üreteç
class PConvUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, ngf=64):
        super(PConvUNet, self).__init__()

        # Encoder
        self.enc1 = PartialConv2d(in_channels, ngf, kernel_size=7, stride=2, padding=3, bias=False) # 512 -> 256
        self.enc2 = PartialConv2d(ngf, ngf*2, kernel_size=5, stride=2, padding=2, bias=False) # 256 -> 128
        self.enc3 = PartialConv2d(ngf*2, ngf*4, kernel_size=5, stride=2, padding=2, bias=False) # 128 -> 64
        self.enc4 = PartialConv2d(ngf*4, ngf*8, kernel_size=3, stride=2, padding=1, bias=False) # 64 -> 32
        self.enc5 = PartialConv2d(ngf*8, ngf*8, kernel_size=3, stride=2, padding=1, bias=False) # 32 -> 16
        self.enc6 = PartialConv2d(ngf*8, ngf*8, kernel_size=3, stride=2, padding=1, bias=False) # 16 -> 8
        self.enc7 = PartialConv2d(ngf*8, ngf*8, kernel_size=3, stride=2, padding=1, bias=False) # 8 -> 4
        self.enc8 = PartialConv2d(ngf*8, ngf*8, kernel_size=3, stride=2, padding=1, bias=False) # 4 -> 2 (Latent space is 2x2, max downsampling 8 times)
        
        # Decoder
        self.dec8 = PartialConv2d(ngf*8 + ngf*8, ngf*8, kernel_size=3, stride=1, padding=1, bias=False)
        self.dec7 = PartialConv2d(ngf*8 + ngf*8, ngf*8, kernel_size=3, stride=1, padding=1, bias=False)
        self.dec6 = PartialConv2d(ngf*8 + ngf*8, ngf*8, kernel_size=3, stride=1, padding=1, bias=False)
        self.dec5 = PartialConv2d(ngf*8 + ngf*8, ngf*8, kernel_size=3, stride=1, padding=1, bias=False)
        self.dec4 = PartialConv2d(ngf*8 + ngf*4, ngf*4, kernel_size=3, stride=1, padding=1, bias=False)
        self.dec3 = PartialConv2d(ngf*4 + ngf*2, ngf*2, kernel_size=3, stride=1, padding=1, bias=False)
        self.dec2 = PartialConv2d(ngf*2 + ngf, ngf, kernel_size=3, stride=1, padding=1, bias=False)
        # The last decoder layer 'dec1' should take the concatenated feature map and mask,
        # and output the final image. The input to dec1 for concatenation comes from enc1 output and the current decoder feature.
        self.dec1 = PartialConv2d(ngf + in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.sigmoid = nn.Sigmoid() # For output mask
        self.tanh = nn.Tanh() # For image output [-1, 1]

    def forward(self, x, mask):
        # x: masked image (input)
        # mask: 0 for masked, 1 for unmasked

        # Store initial input and mask for the last skip connection if needed
        initial_x = x
        initial_mask = mask

        # Encoder
        x1, mask1 = self.enc1(x, mask)
        x1 = self.lrelu(x1)

        x2, mask2 = self.enc2(x1, mask1)
        x2 = self.lrelu(x2)

        x3, mask3 = self.enc3(x2, mask2)
        x3 = self.lrelu(x3)

        x4, mask4 = self.enc4(x3, mask3)
        x4 = self.lrelu(x4)

        x5, mask5 = self.enc5(x4, mask4)
        x5 = self.lrelu(x5)

        x6, mask6 = self.enc6(x5, mask5)
        x6 = self.lrelu(x6)

        x7, mask7 = self.enc7(x6, mask6)
        x7 = self.lrelu(x7)
        
        x8, mask8 = self.enc8(x7, mask7)
        x8 = self.lrelu(x8)

        # Decoder with Skip Connections and Upsampling
        # Dec 8
        x8 = F.interpolate(x8, scale_factor=2, mode='nearest')
        mask8 = F.interpolate(mask8, scale_factor=2, mode='nearest')
        x = torch.cat((x8, x7), 1)
        mask = torch.cat((mask8, mask7), 1)
        x, mask = self.dec8(x, mask)
        x = self.lrelu(x)

        # Dec 7
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        mask = F.interpolate(mask, scale_factor=2, mode='nearest')
        x = torch.cat((x, x6), 1)
        mask = torch.cat((mask, mask6), 1)
        x, mask = self.dec7(x, mask)
        x = self.lrelu(x)

        # Dec 6
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        mask = F.interpolate(mask, scale_factor=2, mode='nearest')
        x = torch.cat((x, x5), 1)
        mask = torch.cat((mask, mask5), 1)
        x, mask = self.dec6(x, mask)
        x = self.lrelu(x)

        # Dec 5
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        mask = F.interpolate(mask, scale_factor=2, mode='nearest')
        x = torch.cat((x, x4), 1)
        mask = torch.cat((mask, mask4), 1)
        x, mask = self.dec5(x, mask)
        x = self.lrelu(x)

        # Dec 4
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        mask = F.interpolate(mask, scale_factor=2, mode='nearest')
        x = torch.cat((x, x3), 1)
        mask = torch.cat((mask, mask3), 1)
        x, mask = self.dec4(x, mask)
        x = self.lrelu(x)

        # Dec 3
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        mask = F.interpolate(mask, scale_factor=2, mode='nearest')
        x = torch.cat((x, x2), 1)
        mask = torch.cat((mask, mask2), 1)
        x, mask = self.dec3(x, mask)
        x = self.lrelu(x)

        # Dec 2
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        mask = F.interpolate(mask, scale_factor=2, mode='nearest')
        x = torch.cat((x, x1), 1)
        mask = torch.cat((mask, mask1), 1)
        x, mask = self.dec2(x, mask)
        x = self.lrelu(x)

        # Final Layer (Dec 1) - Concatenate with the original input at its resolution
        # The input to dec1 should be the feature from the previous decoder step (ngf)
        # and the initial input image itself (in_channels=3)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        mask = F.interpolate(mask, scale_factor=2, mode='nearest')
        
        # Original input x and mask need to be used here.
        # Ensure they are at the correct scale if direct concatenation is intended.
        # For a U-Net, the final output layer takes the features and produces the image.
        # The skip connection would have brought 'information' from the input.
        # If we concatenate the initial input 'x' again, it means the model
        # directly sees the masked input along with its learned features.
        # This is often done, but the channels need to match.
        # The previous dec2 output `x` is `ngf` channels. `initial_x` is `in_channels` (3).
        # So `ngf + in_channels` is correct.
        
        x = torch.cat((x, initial_x), 1)
        mask = torch.cat((mask, initial_mask.expand_as(initial_x)), 1) # Expand mask to 3 channels for concatenation

        x, _ = self.dec1(x, mask) # mask is used inside PConv for calculation
        output = self.tanh(x)

        return output

# Diskriminatör Model (PatchGAN veya ESRGAN benzeri)
class Discriminator(nn.Module):
    def __init__(self, input_channels=3, ndf=64):
        super(Discriminator, self).__init__()

        def conv_block_d(in_f, out_f, stride, use_bn=True):
            layers = [
                nn.Conv2d(in_f, out_f, kernel_size=4, stride=stride, padding=1, bias=False),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            if use_bn:
                layers.append(nn.BatchNorm2d(out_f))
            return nn.Sequential(*layers)
        
        self.main = nn.Sequential(
            # Input: 3 x 512 x 512
            nn.Conv2d(input_channels, ndf, kernel_size=4, stride=2, padding=1, bias=False), # 256x256
            nn.LeakyReLU(0.2, inplace=True),

            conv_block_d(ndf, ndf*2, stride=2), # 128x128
            conv_block_d(ndf*2, ndf*4, stride=2), # 64x64
            conv_block_d(ndf*4, ndf*8, stride=2), # 32x32
            conv_block_d(ndf*8, ndf*8, stride=2), # 16x16
            conv_block_d(ndf*8, ndf*8, stride=2), # 8x8
            conv_block_d(ndf*8, ndf*8, stride=2), # 4x4
            
            # Final output layer
            nn.Conv2d(ndf*8, 1, kernel_size=4, stride=1, padding=1, bias=False) # 1x1 output (effectively a patch)
        )

    def forward(self, x):
        return self.main(x).view(x.size(0), -1) # Flatten to (batch_size, 1)

# --- 4. VGG Özellik Çıkarıcı (Algısal Kayıp İçin) ---
class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super(VGGFeatureExtractor, self).__init__()
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        for param in vgg.parameters():
            param.requires_grad = False
        
        # Use features up to 'block5_conv4' (index 34 in vgg.features for VGG19, which is the Conv2d layer before ReLU)
        # Often, inpainting uses a slightly earlier layer for less semantic features, e.g., 'block4_conv4' (index 27)
        # Let's use block4_conv4 for inpainting, as it captures more texture.
        self.features = nn.Sequential(*list(vgg.features)[:28]) # Adjusted to take features BEFORE ReLU of block4_conv4

    def preprocess_vgg(self, image):
        # Image is already [-1, 1] from generator. Convert to [0, 1] then normalize for VGG.
        image = (image + 1) / 2.0  # Convert from [-1, 1] to [0, 1]
        # VGG preprocessing parameters (mean and std for ImageNet)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(image.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(image.device)
        image = (image - mean) / std
        return image

    def forward(self, x):
        x = self.preprocess_vgg(x)
        return self.features(x)

# --- 5. Kayıp Fonksiyonları ---

def discriminator_loss_ragan(real_output, fake_output):
    # D_loss = BCE(D(x_real) - E[D(x_fake)], 1) + BCE(D(x_fake) - E[D(x_real)], 0)
    loss_real = F.binary_cross_entropy_with_logits(real_output - torch.mean(fake_output), torch.ones_like(real_output))
    loss_fake = F.binary_cross_entropy_with_logits(fake_output - torch.mean(real_output), torch.zeros_like(fake_output))
    return (loss_real + loss_fake) / 2

def generator_loss_ragan(real_output, fake_output):
    # G_loss = BCE(D(x_fake) - E[D(x_real)], 1) + BCE(D(x_real) - E[D(x_fake)], 0)
    loss_fake = F.binary_cross_entropy_with_logits(fake_output - torch.mean(real_output), torch.ones_like(fake_output))
    loss_real = F.binary_cross_entropy_with_logits(real_output - torch.mean(fake_output), torch.zeros_like(real_output))
    return (loss_fake + loss_real) / 2

# İçerik Kaybı (Algısal Kayıp)
def content_loss(hr_features, sr_features):
    return F.l1_loss(hr_features, sr_features) # L1 loss for content

# --- 6. InpaintingTrainer Sınıfı ---

class InpaintingTrainer:
    def __init__(self, generator, discriminator, vgg_feature_extractor, lambda_pixel=1.0, lambda_content=1.0, lambda_adv=0.01):
        self.generator = generator
        self.discriminator = discriminator
        self.vgg_feature_extractor = vgg_feature_extractor
        self.lambda_pixel = lambda_pixel
        self.lambda_content = lambda_content
        self.lambda_adv = lambda_adv

        self.mae_loss = nn.L1Loss()

        self.generator.to(DEVICE)
        self.discriminator.to(DEVICE)
        self.vgg_feature_extractor.to(DEVICE)

        self.g_optimizer = None
        self.d_optimizer = None

        # Metrics for logging
        self.d_losses = []
        self.g_losses = []
        self.content_losses = []
        self.adv_losses = []
        self.pixel_losses = []
        
        self.d_loss_batch_avg = 0.0
        self.g_loss_batch_avg = 0.0
        self.content_loss_batch_avg = 0.0
        self.adv_loss_batch_avg = 0.0
        self.pixel_loss_batch_avg = 0.0
        self.step_count = 0

    def compile(self, g_optimizer_class, d_optimizer_class, lr_g=1e-4, lr_d=1e-4):
        self.g_optimizer = g_optimizer_class(self.generator.parameters(), lr=lr_g, betas=(0.9, 0.999))
        self.d_optimizer = d_optimizer_class(self.discriminator.parameters(), lr=lr_d, betas=(0.9, 0.999))

    def reset_batch_metrics(self):
        self.d_loss_batch_avg = 0.0
        self.g_loss_batch_avg = 0.0
        self.content_loss_batch_avg = 0.0
        self.adv_loss_batch_avg = 0.0
        self.pixel_loss_batch_avg = 0.0
        self.step_count = 0

    def update_batch_metrics(self, d_loss, g_loss, c_loss, adv_loss, p_loss):
        self.d_loss_batch_avg += d_loss
        self.g_loss_batch_avg += g_loss
        self.content_loss_batch_avg += c_loss
        self.adv_loss_batch_avg += adv_loss
        self.pixel_loss_batch_avg += p_loss
        self.step_count += 1

    def save_epoch_metrics(self):
        if self.step_count > 0:
            self.d_losses.append(self.d_loss_batch_avg / self.step_count)
            self.g_losses.append(self.g_loss_batch_avg / self.step_count)
            self.content_losses.append(self.content_loss_batch_avg / self.step_count)
            self.adv_losses.append(self.adv_loss_batch_avg / self.step_count)
            self.pixel_losses.append(self.pixel_loss_batch_avg / self.step_count)
        self.reset_batch_metrics() # Reset for next epoch

    def plot_losses(self, epoch, save_path="./inpainting_results"):
        epochs_ran = len(self.g_losses)
        if epochs_ran == 0:
            print("Henüz kaydedilmiş metrik bulunmuyor.")
            return

        fig, axes = plt.subplots(3, 2, figsize=(15, 18))
        fig.suptitle(f"Eğitim Metrikleri - Epoch {epochs_ran}", fontsize=16)

        def format_epoch_axis(ax):
            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            ax.set_xlabel("Epoch")

        axes[0, 0].plot(range(1, epochs_ran + 1), self.g_losses, label="Jeneratör Kaybı", color='blue')
        axes[0, 0].set_title("Jeneratör Kaybı")
        axes[0, 0].set_ylabel("Kayıp Değeri")
        axes[0, 0].legend()
        format_epoch_axis(axes[0, 0])

        axes[0, 1].plot(range(1, epochs_ran + 1), self.d_losses, label="Diskriminatör Kaybı", color='red')
        axes[0, 1].set_title("Diskriminatör Kaybı")
        axes[0, 1].set_ylabel("Kayıp Değeri")
        axes[0, 1].legend()
        format_epoch_axis(axes[0, 1])

        axes[1, 0].plot(range(1, epochs_ran + 1), self.content_losses, label="Algısal Kayıp", color='green')
        axes[1, 0].set_title("Algısal Kayıp")
        axes[1, 0].set_ylabel("Kayıp Değeri")
        axes[1, 0].legend()
        format_epoch_axis(axes[1, 0])

        axes[1, 1].plot(range(1, epochs_ran + 1), self.adv_losses, label="Adversaryal Kayıp", color='purple')
        axes[1, 1].set_title("Adversaryal Kayıp")
        axes[1, 1].set_ylabel("Kayıp Değeri")
        axes[1, 1].legend()
        format_epoch_axis(axes[1, 1])

        axes[2, 0].plot(range(1, epochs_ran + 1), self.pixel_losses, label="Piksel Kaybı", color='orange')
        axes[2, 0].set_title("Piksel Kaybı")
        axes[2, 0].set_ylabel("Kayıp Değeri")
        axes[2, 0].legend()
        format_epoch_axis(axes[2, 0])
        
        # Boş subplot'ı gizle
        fig.delaxes(axes[2, 1])

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.savefig(os.path.join(save_path, f"loss_curves_epoch_{epoch+1}.png"))
        plt.close(fig)

    def save_sample_images(self, masked_images, original_images, completed_images, epoch, save_path="./inpainting_results/samples"):
        os.makedirs(save_path, exist_ok=True)
        num_samples = min(masked_images.shape[0], 4) # Sadece birkaç örnek kaydet

        fig, axes = plt.subplots(num_samples, 3, figsize=(9, num_samples * 3))
        fig.suptitle(f"Epoch {epoch+1} Örnek Tamamlama Sonuçları", fontsize=16)

        for i in range(num_samples):
            masked_img = denormalize_img_from_neg1_1(masked_images[i].cpu().detach()).permute(1, 2, 0).numpy().astype(np.uint8)
            original_img = denormalize_img_from_neg1_1(original_images[i].cpu().detach()).permute(1, 2, 0).numpy().astype(np.uint8)
            completed_img = denormalize_img_from_neg1_1(completed_images[i].cpu().detach()).permute(1, 2, 0).numpy().astype(np.uint8)

            axes[i, 0].imshow(masked_img)
            axes[i, 0].set_title("Maskeli")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(completed_img)
            axes[i, 1].set_title("Tamamlanmış")
            axes[i, 1].axis("off")

            axes[i, 2].imshow(original_img)
            axes[i, 2].set_title("Orijinal")
            axes[i, 2].axis("off")

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.savefig(os.path.join(save_path, f"samples_epoch_{epoch+1}.png"))
        plt.close(fig)

    def train_step(self, masked_images, original_images, masks):
        masked_images = masked_images.to(DEVICE)
        original_images = original_images.to(DEVICE)
        masks = masks.to(DEVICE)

        # --- Ayırt Edici (Discriminator) Eğitimi ---
        self.d_optimizer.zero_grad()

        # Generate fake completed images
        completed_images_g = self.generator(masked_images, masks).detach() # Detach for D training

        # Discriminator predictions
        real_output = self.discriminator(original_images)
        fake_output = self.discriminator(completed_images_g)

        # Discriminator loss (RaGAN)
        d_loss = discriminator_loss_ragan(real_output, fake_output)

        d_loss.backward()
        self.d_optimizer.step()

        # --- Üreteç (Generator) Eğitimi ---
        self.g_optimizer.zero_grad()

        # Generate fake completed images again (not detached for G training)
        completed_images = self.generator(masked_images, masks)

        # Discriminator prediction on fake images for G loss
        fake_output_for_g = self.discriminator(completed_images)
        real_output_for_g = self.discriminator(original_images) # Need real output for RaGAN G loss

        # Adversarial loss (RaGAN)
        adversarial_loss = generator_loss_ragan(real_output_for_g, fake_output_for_g)

        # Content loss (Perceptual Loss)
        original_features = self.vgg_feature_extractor(original_images)
        completed_features = self.vgg_feature_extractor(completed_images)
        c_loss = content_loss(original_features, completed_features)

        # Pixel loss (L1 Loss)
        # Sadece maskelenen bölgeler üzerinde piksel kaybı hesaplamak da yaygındır:
        # p_loss = self.mae_loss(completed_images * masks, original_images * masks)
        # Ancak genellikle tüm resim üzerinde de hesaplanır:
        p_loss = self.mae_loss(completed_images, original_images)

        # Total Generator loss
        g_loss = (self.lambda_pixel * p_loss) + (self.lambda_content * c_loss) + (self.lambda_adv * adversarial_loss)

        g_loss.backward()
        self.g_optimizer.step()
        
        # Update batch metrics
        self.update_batch_metrics(d_loss.item(), g_loss.item(), c_loss.item(), adversarial_loss.item(), p_loss.item())

        return {
            "d_loss": d_loss.item(),
            "g_loss": g_loss.item(),
            "content_loss": c_loss.item(),
            "adv_loss": adversarial_loss.item(),
            "pixel_loss": p_loss.item(),
        }, completed_images

    def fit(self, dataloader, epochs):
        for epoch in range(epochs):
            self.reset_batch_metrics()
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # İlk batch'i al, görselleştirme için kullanmak üzere sakla
            first_masked_images = None
            first_original_images = None
            first_completed_images = None

            for batch_idx, (masked_images, original_images, masks) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
                metrics, completed_images = self.train_step(masked_images, original_images, masks)

                if batch_idx == 0: # Sadece ilk batch'ten örnekleri al
                    first_masked_images = masked_images
                    first_original_images = original_images
                    first_completed_images = completed_images

            self.save_epoch_metrics() # Save average metrics for the epoch
            self.plot_losses(epoch) # Plot and save loss curves
            self.save_sample_images(first_masked_images, first_original_images, first_completed_images, epoch) # Save sample images
            
            epoch_avg_metrics = self.get_metrics_results_epoch_end() # Get the last saved epoch metrics
            tqdm.write(f"Epoch {epoch+1} average: "
                      f"D_loss: {epoch_avg_metrics['d_loss']:.4f}, G_loss: {epoch_avg_metrics['g_loss']:.4f}, "
                      f"Content_loss: {epoch_avg_metrics['content_loss']:.4f}, Adv_loss: {epoch_avg_metrics['adv_loss']:.4f}, "
                      f"Pixel_loss: {epoch_avg_metrics['pixel_loss']:.4f}")
    
    def get_metrics_results_epoch_end(self):
        # Returns the last calculated epoch-average metrics
        return {
            "d_loss": self.d_losses[-1] if self.d_losses else 0,
            "g_loss": self.g_losses[-1] if self.g_losses else 0,
            "content_loss": self.content_losses[-1] if self.content_losses else 0,
            "adv_loss": self.adv_losses[-1] if self.adv_losses else 0,
            "pixel_loss": self.pixel_losses[-1] if self.pixel_losses else 0,
        }

# --- 7. Ana Çalıştırma Bloğu ---

if __name__ == "__main__":
    # Model Oluşturma
    generator = PConvUNet().to(DEVICE)
    print("Üreteç (PConvUNet) oluşturuldu.")

    discriminator = Discriminator().to(DEVICE)
    print("Ayırt Edici oluşturuldu.")

    # VGG feature extractor
    vgg_feature_extractor = VGGFeatureExtractor().to(DEVICE)
    print("VGG Feature Extractor oluşturuldu.")

    # Eğitim Veri Setini Oluşturma
    train_dataloader = create_dataloader(BASE_DATASET_DIR, BATCH_SIZE, NUM_WORKERS, IMG_SIZE)

    if train_dataloader is None:
        print("Veri seti oluşturulamadı. Lütfen dizin yollarını ve görüntülerin varlığını kontrol edin.")
        print(f"Örnek veri setiniz '{BASE_DATASET_DIR}' içinde 512x512'ye yakın boyutlarda PNG/JPG vb. resimler içermelidir.")
        exit()
    else:
        print("\nVeri seti başarıyla oluşturuldu. Eğitime hazırız.")
        
        # İlk batch'i kontrol edelim
        for masked_batch_sample, original_batch_sample, mask_batch_sample in train_dataloader:
            print(f"Örnek Maskeli Görüntü Batch Şekli: {masked_batch_sample.shape}")
            print(f"Örnek Orijinal Görüntü Batch Şekli: {original_batch_sample.shape}")
            print(f"Örnek Maske Batch Şekli: {mask_batch_sample.shape}")
            break

        # InpaintingTrainer Nesnesini Oluşturma
        inpainting_trainer = InpaintingTrainer(
            generator=generator,
            discriminator=discriminator,
            vgg_feature_extractor=vgg_feature_extractor,
            lambda_pixel=LAMBDA_PIXEL,
            lambda_content=LAMBDA_CONTENT,
            lambda_adv=LAMBDA_ADV
        )

        # InpaintingTrainer'ı derle
        inpainting_trainer.compile(
            g_optimizer_class=optim.Adam,
            d_optimizer_class=optim.Adam,
            lr_g=LEARNING_RATE_G,
            lr_d=LEARNING_RATE_D
        )
        print("Eğitimci derlendi. Optimizasyonlar ayarlandı.")

        # Eğitimi başlat
        print(f"\nEğitim {EPOCHS} epoch boyunca başlayacak...")
        inpainting_trainer.fit(train_dataloader, EPOCHS)
        print("\nEğitim tamamlandı!")

        # İsteğe bağlı: Modeli kaydet
        torch.save(generator.state_dict(), os.path.join("./inpainting_results", "generator_final.pth"))
        torch.save(discriminator.state_dict(), os.path.join("./inpainting_results", "discriminator_final.pth"))
        print("Model ağırlıkları kaydedildi.")