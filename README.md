# Partial Convolution İle Image Inpainting Modeli - Eğitim Rehberi

## 📋 İçindekiler
1. [Model Hakkında](#model-hakkında)
2. [Gereksinimler](#gereksinimler)
3. [Veri Seti Hazırlığı](#veri-seti-hazırlığı)
4. [Model Mimarisi](#model-mimarisi)
5. [Eğitim Parametreleri](#eğitim-parametreleri)
6. [Kullanım](#kullanım)
7. [Çıktılar](#çıktılar)
8. [Sorun Giderme](#sorun-giderme)

## 🎯 Model Hakkında

Bu model, **Partial Convolution** tabanlı bir U-Net mimarisi kullanarak görüntü tamamlama (image inpainting) işlemi gerçekleştirir. Model aşağıdaki özelliklere sahiptir:

- **PConvUNet Generator**: Partial Convolution katmanları ile eksik bölgeleri tamamlar
- **Discriminator**: Gerçek ve sahte görüntüleri ayırt eder (RaGAN kayıp fonksiyonu)
- **VGG Feature Extractor**: Perceptual loss hesaplar
- **Multi-scale Loss**: Piksel, algısal ve adversarial kayıp kombinasyonu

### 🔧 Temel Özellikler
- 512x512 çözünürlük desteği
- Düzensiz (irregular) ve dikdörtgen maske türleri
- Otomatik maske oluşturma
- Gerçek zamanlı eğitim takibi
- Görsel sonuç raporlama

## 📦 Gereksinimler

### Python Kütüphaneleri
```bash
pip install torch torchvision
pip install pillow numpy opencv-python
pip install matplotlib tqdm glob2
```

### Sistem Gereksinimleri
- **GPU**: CUDA destekli (önerilen)
- **RAM**: En az 8GB
- **VRAM**: En az 8GB (512x512 için)
- **Depolama**: Veri seti için yeterli alan

## 📁 Veri Seti Hazırlığı

### Klasör Yapısı
```
projeniz/
├── data/              # Eğitim görüntüleri
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
├── inpainting_results/  # Otomatik oluşturulur
└── model_script.py
```

### Desteklenen Formatlar
- PNG, JPG, JPEG, BMP, TIFF
- Minimum 512x512 çözünürlük önerilir
- RGB renk formatı

### Önerilen Veri Setleri
- **CelebA-HQ**: Yüz görüntüleri
- **Places365**: Manzara görüntüleri
- **DIV2K**: Yüksek çözünürlüklü genel görüntüler

## 🏗️ Model Mimarisi

### Generator (PConvUNet)
```
Encoder:  Conv7→Conv5→Conv5→Conv3→Conv3→Conv3→Conv3→Conv3
          512→256→128→64→32→16→8→4→2

Decoder:  Upsampling + Skip Connections
          2→4→8→16→32→64→128→256→512
```

### Discriminator (PatchGAN)
```
Input: 3×512×512
↓ Conv4×4, stride=2, LeakyReLU
Output: 1×1 (Patch prediction)
```

### Kayıp Fonksiyonları
1. **Pixel Loss (L1)**: `λ_pixel = 1.0`
2. **Perceptual Loss (VGG)**: `λ_content = 1.0`  
3. **Adversarial Loss (RaGAN)**: `λ_adv = 0.01`

## ⚙️ Eğitim Parametreleri

### Varsayılan Ayarlar
```python
BATCH_SIZE = 16        # GPU belleğine göre ayarlayın
NUM_WORKERS = 4        # CPU core sayısı
EPOCHS = 8             # Eğitim epoch sayısı
IMG_SIZE = (512, 512)  # Görüntü boyutu

LEARNING_RATE_G = 1e-4  # Generator öğrenme oranı
LEARNING_RATE_D = 1e-4  # Discriminator öğrenme oranı
```

### Maske Parametreleri
```python
mask_type = "irregular"    # "rectangle" veya "irregular"
max_masks = 3             # Maksimum maske sayısı
max_size_ratio = 0.3      # Maksimum maske boyut oranı
max_thickness = 20        # Maske kalınlığı
```

## 🚀 Kullanım

### 1. Veri Seti Yolunu Ayarlayın
```python
BASE_DATASET_DIR = "./data"  # Kendi veri setinizin yolu
```

### 2. Modeli Çalıştırın
```bash
python model_script.py
```

### 3. Eğitim Süreci
Model otomatik olarak:
- Veri setini yükler
- Rastgele maskeler oluşturur
- Generator ve Discriminator'ı eğitir
- Her epoch'ta örnek sonuçlar kaydeder
- Kayıp grafikleri oluşturur

## 📊 Çıktılar

### Klasör Yapısı (Eğitim Sonrası)
```
inpainting_results/
├── samples/                    # Örnek tamamlama sonuçları
│   ├── samples_epoch_1.png
│   ├── samples_epoch_2.png
│   └── ...
├── loss_curves_epoch_1.png     # Kayıp grafikleri
├── loss_curves_epoch_2.png
├── generator_final.pth         # Eğitilmiş generator
└── discriminator_final.pth     # Eğitilmiş discriminator
```

### Görsel Çıktılar
- **Örnek Sonuçlar**: Maskeli → Tamamlanmış → Orijinal karşılaştırma
- **Kayıp Grafikleri**: 5 farklı metrik için epoch bazlı grafikler
- **Model Ağırlıkları**: PyTorch `.pth` formatında

## 🔍 Eğitim Takibi

### Konsol Çıktısı
```
Epoch 1/8
100%|██████████| 125/125 [02:34<00:00,  1.23s/it]
Epoch 1 average: D_loss: 0.6891, G_loss: 1.2345, Content_loss: 0.0234, Adv_loss: 0.5678, Pixel_loss: 0.0987
```

### Kayıp Metrikleri
- **D_loss**: Discriminator kaybı
- **G_loss**: Generator toplam kaybı
- **Content_loss**: VGG perceptual kaybı
- **Adv_loss**: Adversarial kayıp
- **Pixel_loss**: L1 piksel kaybı

## 🛠️ Sorun Giderme

### Yaygın Hatalar ve Çözümler

#### 1. CUDA Out of Memory
```python
# Batch size'ı azaltın
BATCH_SIZE = 8  # veya 4

# Görüntü boyutunu küçültün
IMG_SIZE = (256, 256)
```

#### 2. Düşük Performans
```python
# Learning rate ayarlayın
LEARNING_RATE_G = 2e-4
LEARNING_RATE_D = 2e-4

# Epoch sayısını artırın
EPOCHS = 20

# Lambda değerlerini ayarlayın
LAMBDA_PIXEL = 2.0
LAMBDA_CONTENT = 0.5
```

#### 3. Maske Sorunları
```python
# Maske tipini değiştirin
mask_type = "rectangle"  # Daha basit maskeler

# Maske boyutunu ayarlayın
max_size_ratio = 0.2  # Daha küçük maskeler
```

### Performans Optimizasyonu

#### GPU Kullanımı
```python
# Multi-GPU desteği için
if torch.cuda.device_count() > 1:
    generator = nn.DataParallel(generator)
    discriminator = nn.DataParallel(discriminator)
```

#### Bellek Optimizasyonu
```python
# Gradient accumulation
accumulation_steps = 4
BATCH_SIZE = BATCH_SIZE // accumulation_steps
```

## 📈 Model Değerlendirme

### Kalite Metrikleri
- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index
- **LPIPS**: Learned Perceptual Image Patch Similarity

### Test Etme
```python
# Eğitilmiş modeli yükle
generator.load_state_dict(torch.load("./inpainting_results/generator_final.pth"))
generator.eval()

# Test görüntüsü ile tamamlama yap
with torch.no_grad():
    completed = generator(masked_image, mask)
```

## 🔄 Model Geliştirme

### Hiperparametre Ayarlama
- Learning rate schedules
- Farklı kayıp ağırlıkları
- Maske çeşitliliği
- Augmentasyon teknikleri

## 📝 Lisans ve Kullanım

Bu model akademik ve araştırma amaçlı kullanım için tasarlanmıştır. Ticari kullanım öncesinde ilgili makalelerin lisans koşullarını kontrol ediniz.

---

**Not**: Bu model, GPU desteği ile en iyi performansı gösterir. CPU üzerinde eğitim oldukça yavaş olacaktır.