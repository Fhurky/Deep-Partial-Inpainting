import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
import threading
from torchvision import transforms

# Eğitim kodunuzdaki model tanımlarını burada da kullanacağız
class PartialConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(PartialConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                            padding, dilation, groups, bias)
        self.mask_kernel = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])
        self.slide_winsize = self.kernel_size[0] * self.kernel_size[1]
        
    def forward(self, input, mask):
        device = input.device
        self.mask_kernel = self.mask_kernel.to(device)
        
        single_channel_mask_for_conv = mask[:, 0:1, :, :]
        
        with torch.no_grad():
            update_mask = F.conv2d(single_channel_mask_for_conv, self.mask_kernel, bias=None, stride=self.stride,
                                   padding=self.padding, dilation=self.dilation, groups=1)
        
        mask_ratio = self.slide_winsize / (update_mask + 1e-8)
        update_mask_for_next_layer = torch.clamp(update_mask, 0, 1)
        output_features = super(PartialConv2d, self).forward(input * mask)
        output_features = output_features * mask_ratio.expand_as(output_features)
        
        return output_features, update_mask_for_next_layer.expand(-1, self.out_channels, -1, -1)

class PConvUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, ngf=64):
        super(PConvUNet, self).__init__()

        # Encoder
        self.enc1 = PartialConv2d(in_channels, ngf, kernel_size=7, stride=2, padding=3, bias=False)
        self.enc2 = PartialConv2d(ngf, ngf*2, kernel_size=5, stride=2, padding=2, bias=False)
        self.enc3 = PartialConv2d(ngf*2, ngf*4, kernel_size=5, stride=2, padding=2, bias=False)
        self.enc4 = PartialConv2d(ngf*4, ngf*8, kernel_size=3, stride=2, padding=1, bias=False)
        self.enc5 = PartialConv2d(ngf*8, ngf*8, kernel_size=3, stride=2, padding=1, bias=False)
        self.enc6 = PartialConv2d(ngf*8, ngf*8, kernel_size=3, stride=2, padding=1, bias=False)
        self.enc7 = PartialConv2d(ngf*8, ngf*8, kernel_size=3, stride=2, padding=1, bias=False)
        self.enc8 = PartialConv2d(ngf*8, ngf*8, kernel_size=3, stride=2, padding=1, bias=False)
        
        # Decoder
        self.dec8 = PartialConv2d(ngf*8 + ngf*8, ngf*8, kernel_size=3, stride=1, padding=1, bias=False)
        self.dec7 = PartialConv2d(ngf*8 + ngf*8, ngf*8, kernel_size=3, stride=1, padding=1, bias=False)
        self.dec6 = PartialConv2d(ngf*8 + ngf*8, ngf*8, kernel_size=3, stride=1, padding=1, bias=False)
        self.dec5 = PartialConv2d(ngf*8 + ngf*8, ngf*8, kernel_size=3, stride=1, padding=1, bias=False)
        self.dec4 = PartialConv2d(ngf*8 + ngf*4, ngf*4, kernel_size=3, stride=1, padding=1, bias=False)
        self.dec3 = PartialConv2d(ngf*4 + ngf*2, ngf*2, kernel_size=3, stride=1, padding=1, bias=False)
        self.dec2 = PartialConv2d(ngf*2 + ngf, ngf, kernel_size=3, stride=1, padding=1, bias=False)
        self.dec1 = PartialConv2d(ngf + in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, x, mask):
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

        # Decoder with Skip Connections
        x8 = F.interpolate(x8, scale_factor=2, mode='nearest')
        mask8 = F.interpolate(mask8, scale_factor=2, mode='nearest')
        x = torch.cat((x8, x7), 1)
        mask = torch.cat((mask8, mask7), 1)
        x, mask = self.dec8(x, mask)
        x = self.lrelu(x)

        x = F.interpolate(x, scale_factor=2, mode='nearest')
        mask = F.interpolate(mask, scale_factor=2, mode='nearest')
        x = torch.cat((x, x6), 1)
        mask = torch.cat((mask, mask6), 1)
        x, mask = self.dec7(x, mask)
        x = self.lrelu(x)

        x = F.interpolate(x, scale_factor=2, mode='nearest')
        mask = F.interpolate(mask, scale_factor=2, mode='nearest')
        x = torch.cat((x, x5), 1)
        mask = torch.cat((mask, mask5), 1)
        x, mask = self.dec6(x, mask)
        x = self.lrelu(x)

        x = F.interpolate(x, scale_factor=2, mode='nearest')
        mask = F.interpolate(mask, scale_factor=2, mode='nearest')
        x = torch.cat((x, x4), 1)
        mask = torch.cat((mask, mask4), 1)
        x, mask = self.dec5(x, mask)
        x = self.lrelu(x)

        x = F.interpolate(x, scale_factor=2, mode='nearest')
        mask = F.interpolate(mask, scale_factor=2, mode='nearest')
        x = torch.cat((x, x3), 1)
        mask = torch.cat((mask, mask3), 1)
        x, mask = self.dec4(x, mask)
        x = self.lrelu(x)

        x = F.interpolate(x, scale_factor=2, mode='nearest')
        mask = F.interpolate(mask, scale_factor=2, mode='nearest')
        x = torch.cat((x, x2), 1)
        mask = torch.cat((mask, mask2), 1)
        x, mask = self.dec3(x, mask)
        x = self.lrelu(x)

        x = F.interpolate(x, scale_factor=2, mode='nearest')
        mask = F.interpolate(mask, scale_factor=2, mode='nearest')
        x = torch.cat((x, x1), 1)
        mask = torch.cat((mask, mask1), 1)
        x, mask = self.dec2(x, mask)
        x = self.lrelu(x)

        x = F.interpolate(x, scale_factor=2, mode='nearest')
        mask = F.interpolate(mask, scale_factor=2, mode='nearest')
        x = torch.cat((x, initial_x), 1)
        mask = torch.cat((mask, initial_mask.expand_as(initial_x)), 1)
        x, _ = self.dec1(x, mask)
        output = self.tanh(x)

        return output

class InpaintingGUI:
    def __init__(self):
        # Tema ve görünüm ayarları
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Ana pencere
        self.root = ctk.CTk()
        self.root.title("AI Image Inpainting - Görüntü Tamamlama")
        self.root.geometry("1400x900")
        self.root.resizable(True, True)
        
        # Değişkenler
        self.original_image = None
        self.display_image = None
        self.mask_image = None
        self.result_image = None
        self.canvas_width = 512
        self.canvas_height = 512
        self.brush_size = 20
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.drawing = False
        
        self.setup_ui()
        self.load_model()
        
    def setup_ui(self):
        # Ana grid layout
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        
        # Sol panel - Kontroller
        self.control_frame = ctk.CTkFrame(self.root, width=300)
        self.control_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.control_frame.grid_propagate(False)
        
        # Başlık
        title_label = ctk.CTkLabel(
            self.control_frame, 
            text="🎨 AI Image Inpainting", 
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title_label.pack(pady=20)
        
        # Model durumu
        self.model_status = ctk.CTkLabel(
            self.control_frame,
            text="Model: Yükleniyor...",
            font=ctk.CTkFont(size=12)
        )
        self.model_status.pack(pady=5)
        
        # Görüntü yükleme butonu
        self.load_btn = ctk.CTkButton(
            self.control_frame,
            text="📁 Görüntü Yükle",
            command=self.load_image,
            width=250,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.load_btn.pack(pady=10)
        
        # Fırça boyutu ayarı
        brush_frame = ctk.CTkFrame(self.control_frame)
        brush_frame.pack(pady=10, padx=20, fill="x")
        
        ctk.CTkLabel(brush_frame, text="Fırça Boyutu:", font=ctk.CTkFont(size=12)).pack(pady=5)
        
        self.brush_slider = ctk.CTkSlider(
            brush_frame,
            from_=5,
            to=50,
            number_of_steps=45,
            command=self.update_brush_size
        )
        self.brush_slider.set(self.brush_size)
        self.brush_slider.pack(pady=5, padx=10, fill="x")
        
        self.brush_label = ctk.CTkLabel(brush_frame, text=f"{self.brush_size}px")
        self.brush_label.pack()
        
        # Maske kontrolleri
        mask_frame = ctk.CTkFrame(self.control_frame)
        mask_frame.pack(pady=10, padx=20, fill="x")
        
        ctk.CTkLabel(mask_frame, text="Maske İşlemleri:", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=5)
        
        self.clear_mask_btn = ctk.CTkButton(
            mask_frame,
            text="🗑️ Maskeyi Temizle",
            command=self.clear_mask,
            width=200,
            height=35
        )
        self.clear_mask_btn.pack(pady=5)
        
        # İnpainting butonu
        self.inpaint_btn = ctk.CTkButton(
            self.control_frame,
            text="✨ Görüntüyü Tamamla",
            command=self.start_inpainting,
            width=250,
            height=50,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color="#2E8B57",
            hover_color="#228B22"
        )
        self.inpaint_btn.pack(pady=20)
        
        # Progress bar
        self.progress = ctk.CTkProgressBar(self.control_frame)
        self.progress.pack(pady=10, padx=20, fill="x")
        self.progress.set(0)
        
        # Kaydetme butonu
        self.save_btn = ctk.CTkButton(
            self.control_frame,
            text="💾 Sonucu Kaydet",
            command=self.save_result,
            width=250,
            height=40,
            state="disabled"
        )
        self.save_btn.pack(pady=10)
        
        # Sağ panel - Canvas ve görüntüler
        self.canvas_frame = ctk.CTkFrame(self.root)
        self.canvas_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        # Canvas başlığı
        canvas_title = ctk.CTkLabel(
            self.canvas_frame,
            text="Görüntü Düzenleme Alanı",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        canvas_title.pack(pady=10)
        
        # Talimatlar
        instructions = ctk.CTkLabel(
            self.canvas_frame,
            text="🖱️ Tamamlanmasını istediğiniz alanları fare ile boyayın",
            font=ctk.CTkFont(size=12)
        )
        instructions.pack(pady=5)
        
        # Canvas
        self.canvas = tk.Canvas(
            self.canvas_frame,
            width=self.canvas_width,
            height=self.canvas_height,
            bg="white",
            cursor="crosshair"
        )
        self.canvas.pack(pady=10)
        
        # Canvas olayları
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)
        
        # Sonuç gösterimi için frame
        self.result_frame = ctk.CTkFrame(self.canvas_frame)
        self.result_frame.pack(pady=10, fill="x")
        
        result_title = ctk.CTkLabel(
            self.result_frame,
            text="Tamamlanmış Görüntü",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        result_title.pack(pady=5)
        
        self.result_canvas = tk.Canvas(
            self.result_frame,
            width=256,
            height=256,
            bg="lightgray"
        )
        self.result_canvas.pack(pady=5)
        
    def load_model(self):
        """Eğitilen modeli yükle"""
        try:
            model_path = "inpainting_results/generator_final.pth"
            if os.path.exists(model_path):
                self.model = PConvUNet()
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.to(self.device)
                self.model.eval()
                self.model_status.configure(text=f"Model: Yüklendi ✅ ({self.device})")
            else:
                self.model_status.configure(text="Model: Bulunamadı ❌")
                messagebox.showwarning("Uyarı", f"Model dosyası bulunamadı: {model_path}")
        except Exception as e:
            self.model_status.configure(text="Model: Hata ❌")
            messagebox.showerror("Hata", f"Model yüklenirken hata: {str(e)}")
    
    def load_image(self):
        """Görüntü yükle"""
        file_path = filedialog.askopenfilename(
            title="Görüntü Seçin",
            filetypes=[
                ("Görüntü Dosyaları", "*.png *.jpg *.jpeg *.bmp *.tiff"),
                ("Tüm Dosyalar", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Görüntüyü yükle ve boyutlandır
                self.original_image = Image.open(file_path).convert("RGB")
                self.original_image = self.original_image.resize((self.canvas_width, self.canvas_height), Image.Resampling.LANCZOS)
                
                # Canvas'a göster
                self.display_image = ImageTk.PhotoImage(self.original_image)
                self.canvas.delete("all")
                self.canvas.create_image(0, 0, anchor="nw", image=self.display_image)
                
                # Maske görüntüsünü sıfırla
                self.mask_image = Image.new("L", (self.canvas_width, self.canvas_height), 0)
                
                # Butonları etkinleştir
                self.clear_mask_btn.configure(state="normal")
                if self.model:
                    self.inpaint_btn.configure(state="normal")
                
            except Exception as e:
                messagebox.showerror("Hata", f"Görüntü yüklenirken hata: {str(e)}")
    
    def update_brush_size(self, value):
        """Fırça boyutunu güncelle"""
        self.brush_size = int(value)
        self.brush_label.configure(text=f"{self.brush_size}px")
    
    def start_draw(self, event):
        """Çizim başlat"""
        if self.original_image:
            self.drawing = True
            self.draw(event)
    
    def draw(self, event):
        """Maske çiz"""
        if self.drawing and self.original_image:
            x, y = event.x, event.y
            
            # Canvas'a kırmızı daire çiz
            r = self.brush_size // 2
            self.canvas.create_oval(
                x - r, y - r, x + r, y + r,
                fill="red", outline="red", width=0,
                tags="mask"
            )
            
            # Maske görüntüsüne de çiz
            draw = ImageDraw.Draw(self.mask_image)
            draw.ellipse([x - r, y - r, x + r, y + r], fill=255)
    
    def stop_draw(self, event):
        """Çizimi durdur"""
        self.drawing = False
    
    def clear_mask(self):
        """Maskeyi temizle"""
        if self.original_image:
            # Canvas'taki maske öğelerini sil
            self.canvas.delete("mask")
            
            # Maske görüntüsünü sıfırla
            self.mask_image = Image.new("L", (self.canvas_width, self.canvas_height), 0)
    
    def start_inpainting(self):
        """İnpainting işlemini başlat"""
        if not self.original_image or not self.model:
            messagebox.showwarning("Uyarı", "Lütfen önce bir görüntü yükleyin ve model yüklendiğinden emin olun.")
            return
        
        # Butonları devre dışı bırak
        self.inpaint_btn.configure(state="disabled")
        self.load_btn.configure(state="disabled")
        
        # Progress bar'ı başlat
        self.progress.set(0)
        
        # Threading ile inpainting işlemini başlat
        thread = threading.Thread(target=self.perform_inpainting)
        thread.daemon = True
        thread.start()
    
    def perform_inpainting(self):
        """İnpainting işlemini gerçekleştir"""
        try:
            # Progress güncelle
            self.root.after(0, lambda: self.progress.set(0.2))
            
            # Görüntü ve maskeyi tensor'a çevir
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            
            # Görüntüyü hazırla
            img_tensor = transform(self.original_image).unsqueeze(0)
            img_tensor = (img_tensor * 2.0) - 1.0  # [-1, 1] normalize
            
            # Maskeyi hazırla (0: maskeli, 1: maskesiz)
            mask_np = np.array(self.mask_image) / 255.0  # [0, 1]
            mask_np = 1.0 - mask_np  # Ters çevir (0: maskeli, 1: maskesiz)
            mask_tensor = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0)
            mask_tensor = mask_tensor.expand(-1, 3, -1, -1)  # 3 kanala genişlet
            
            # GPU'ya taşı
            img_tensor = img_tensor.to(self.device)
            mask_tensor = mask_tensor.to(self.device)
            
            # Maskeli görüntü oluştur
            masked_img = img_tensor * mask_tensor
            
            self.root.after(0, lambda: self.progress.set(0.5))
            
            # İnpainting yap
            with torch.no_grad():
                result_tensor = self.model(masked_img, mask_tensor)
            
            self.root.after(0, lambda: self.progress.set(0.8))
            
            # Sonucu görüntüye çevir
            result_tensor = (result_tensor + 1.0) / 2.0  # [0, 1] denormalize
            result_tensor = torch.clamp(result_tensor, 0, 1)
            
            result_np = result_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            result_np = (result_np * 255).astype(np.uint8)
            
            self.result_image = Image.fromarray(result_np)
            
            # Sonucu göster
            self.root.after(0, self.show_result)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Hata", f"İnpainting sırasında hata: {str(e)}"))
        finally:
            # Butonları yeniden etkinleştir
            self.root.after(0, self.reset_ui)
    
    def show_result(self):
        """Sonucu göster"""
        if self.result_image:
            # Sonuç canvas'ına göster
            result_display = self.result_image.resize((256, 256), Image.Resampling.LANCZOS)
            self.result_photo = ImageTk.PhotoImage(result_display)
            
            self.result_canvas.delete("all")
            self.result_canvas.create_image(0, 0, anchor="nw", image=self.result_photo)
            
            # Kaydet butonunu etkinleştir
            self.save_btn.configure(state="normal")
            
            # Progress tamamla
            self.progress.set(1.0)
    
    def reset_ui(self):
        """UI'yi sıfırla"""
        self.inpaint_btn.configure(state="normal")
        self.load_btn.configure(state="normal")
        self.progress.set(0)
    
    def save_result(self):
        """Sonucu kaydet"""
        if not self.result_image:
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Sonucu Kaydet",
            defaultextension=".png",
            filetypes=[
                ("PNG Dosyaları", "*.png"),
                ("JPEG Dosyaları", "*.jpg"),
                ("Tüm Dosyalar", "*.*")
            ]
        )
        
        if file_path:
            try:
                self.result_image.save(file_path)
                messagebox.showinfo("Başarılı", f"Görüntü kaydedildi: {file_path}")
            except Exception as e:
                messagebox.showerror("Hata", f"Kaydetme sırasında hata: {str(e)}")
    
    def run(self):
        """Uygulamayı çalıştır"""
        self.root.mainloop()

if __name__ == "__main__":
    # Gerekli kütüphanelerin kontrolü
    try:
        import customtkinter
    except ImportError:
        print("CustomTkinter kütüphanesi bulunamadı!")
        print("Yüklemek için: pip install customtkinter")
        exit(1)
    
    # Uygulamayı başlat
    app = InpaintingGUI()
    app.run()