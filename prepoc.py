import os
import cv2

# Kaynak klasör (resimlerin bulunduğu yer)
input_folder = './dataset'  # Örneğin aynı dizinde "resimler" klasörü varsa
# Çıkış klasörü
output_folder = './data'

# Çıkış klasörü yoksa oluştur
os.makedirs(output_folder, exist_ok=True)

# Desteklenen resim uzantıları
image_extensions = ('.jpg', '.jpeg', '.png')

# Resim dosyalarını al
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(image_extensions)]

# Her resmi sırayla okuyup kaydet
for i, filename in enumerate(image_files, start=1):
    image_path = os.path.join(input_folder, filename)
    img = cv2.imread(image_path)
    
    if img is not None:
        output_path = os.path.join(output_folder, f'IMG_{i}.jpg')
        cv2.imwrite(output_path, img)
        print(f"{filename} -> {output_path}")
    else:
        print(f"Resim okunamadı: {filename}")
