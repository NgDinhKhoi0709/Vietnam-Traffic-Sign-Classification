"""
Demo: Chiết xuất và visualize đặc trưng CCV (Color Coherence Vector) từ một ảnh ngẫu nhiên
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from ccv_feature import extract_ccv_features, blur_and_quantize
from scipy import ndimage


def get_random_image_from_train():
    """Lấy ngẫu nhiên một ảnh từ thư mục train"""
    train_path = 'vn-signs/train'
    
    # Lấy danh sách các thư mục con (các class)
    classes = [d for d in os.listdir(train_path) 
               if os.path.isdir(os.path.join(train_path, d))]
    
    # Chọn ngẫu nhiên một class
    random_class = random.choice(classes)
    class_path = os.path.join(train_path, random_class)
    
    # Lấy danh sách các file ảnh trong class đó
    images = [f for f in os.listdir(class_path) 
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Chọn ngẫu nhiên một ảnh
    random_image = random.choice(images)
    image_path = os.path.join(class_path, random_image)
    
    return image_path, random_class


def visualize_ccv_features(image_path):
    """
    Chiết xuất và visualize đặc trưng CCV từ một ảnh
    
    Tham số:
    ----------
    image_path : str
        Đường dẫn đến ảnh
    """
    # Đọc ảnh
    image = cv2.imread(image_path)
    if image is None:
        print(f"Không thể đọc ảnh: {image_path}")
        return
    
    # Chuyển sang RGB để hiển thị đúng màu
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Chiết xuất đặc trưng CCV với các cấu hình khác nhau
    ccv_bgr_8 = extract_ccv_features(image, n_bins=8, threshold=100, color_space='BGR')
    ccv_hsv_8 = extract_ccv_features(image, n_bins=8, threshold=100, color_space='HSV')
    ccv_bgr_16 = extract_ccv_features(image, n_bins=16, threshold=100, color_space='BGR')
    
    # Tạo ảnh đã lượng tử hóa để visualize
    quantized_bgr = blur_and_quantize(image, n_bins=8, blur_size=7, color_space='BGR')
    quantized_hsv = blur_and_quantize(image, n_bins=8, blur_size=7, color_space='HSV')
    
    # Tạo ảnh màu từ quantized để hiển thị
    # Chuyển đổi chỉ số màu thành màu RGB
    n_bins = 8
    quantized_bgr_color = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            idx = quantized_bgr[i, j]
            b = (idx // (n_bins * n_bins)) * (256 // n_bins) + (256 // (2 * n_bins))
            g = ((idx // n_bins) % n_bins) * (256 // n_bins) + (256 // (2 * n_bins))
            r = (idx % n_bins) * (256 // n_bins) + (256 // (2 * n_bins))
            quantized_bgr_color[i, j] = [r, g, b]  # RGB
    
    # Tách coherent và incoherent
    ccv_bgr_coherent = ccv_bgr_8[::2]  # Các phần tử chẵn
    ccv_bgr_incoherent = ccv_bgr_8[1::2]  # Các phần tử lẻ
    
    ccv_hsv_coherent = ccv_hsv_8[::2]
    ccv_hsv_incoherent = ccv_hsv_8[1::2]
    
    # Tạo figure với nhiều subplots
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Hiển thị ảnh gốc
    ax1 = plt.subplot(3, 4, 1)
    ax1.imshow(image_rgb)
    ax1.set_title(f'Ảnh gốc\n{os.path.basename(image_path)}\nKích thước: {image.shape[1]}×{image.shape[0]}', 
                  fontsize=9, fontweight='bold')
    ax1.axis('off')
    
    # 2. Ảnh sau khi lượng tử hóa (BGR)
    ax2 = plt.subplot(3, 4, 2)
    ax2.imshow(quantized_bgr_color)
    ax2.set_title('Ảnh đã lượng tử hóa\n(BGR, 8 bins/channel = 512 màu)', 
                  fontsize=9, fontweight='bold')
    ax2.axis('off')
    
    # 3. CCV BGR (Coherent vs Incoherent)
    ax3 = plt.subplot(3, 4, 3)
    width = 0.35
    x = np.arange(len(ccv_bgr_coherent))
    # Chỉ hiển thị các bin có giá trị > 0 để dễ nhìn
    mask = (ccv_bgr_coherent > 0.001) | (ccv_bgr_incoherent > 0.001)
    if mask.sum() > 0:
        x_masked = x[mask]
        ax3.bar(x_masked - width/2, ccv_bgr_coherent[mask], width, 
                label='Coherent', alpha=0.8, color='green')
        ax3.bar(x_masked + width/2, ccv_bgr_incoherent[mask], width, 
                label='Incoherent', alpha=0.8, color='red')
    ax3.set_title('CCV BGR (8 bins)\nCoherent vs Incoherent', 
                  fontsize=9, fontweight='bold')
    ax3.set_xlabel('Color bin')
    ax3.set_ylabel('Normalized frequency')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 4. CCV BGR Full Vector
    ax4 = plt.subplot(3, 4, 4)
    ax4.bar(range(len(ccv_bgr_8)), ccv_bgr_8, alpha=0.7, color='purple', width=1.0)
    ax4.set_title(f'CCV BGR Full Vector\nShape: {ccv_bgr_8.shape}', 
                  fontsize=9, fontweight='bold')
    ax4.set_xlabel('Feature index')
    ax4.set_ylabel('Value')
    ax4.grid(True, alpha=0.3)
    
    # 5. Ảnh HSV
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ax5 = plt.subplot(3, 4, 5)
    ax5.imshow(image_hsv)
    ax5.set_title('Ảnh trong không gian HSV', fontsize=9, fontweight='bold')
    ax5.axis('off')
    
    # 6. CCV HSV (Coherent vs Incoherent)
    ax6 = plt.subplot(3, 4, 6)
    mask_hsv = (ccv_hsv_coherent > 0.001) | (ccv_hsv_incoherent > 0.001)
    if mask_hsv.sum() > 0:
        x_masked_hsv = x[mask_hsv]
        ax6.bar(x_masked_hsv - width/2, ccv_hsv_coherent[mask_hsv], width, 
                label='Coherent', alpha=0.8, color='green')
        ax6.bar(x_masked_hsv + width/2, ccv_hsv_incoherent[mask_hsv], width, 
                label='Incoherent', alpha=0.8, color='red')
    ax6.set_title('CCV HSV (8 bins)\nCoherent vs Incoherent', 
                  fontsize=9, fontweight='bold')
    ax6.set_xlabel('Color bin')
    ax6.set_ylabel('Normalized frequency')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)
    
    # 7. CCV HSV Full Vector
    ax7 = plt.subplot(3, 4, 7)
    ax7.bar(range(len(ccv_hsv_8)), ccv_hsv_8, alpha=0.7, color='orange', width=1.0)
    ax7.set_title(f'CCV HSV Full Vector\nShape: {ccv_hsv_8.shape}', 
                  fontsize=9, fontweight='bold')
    ax7.set_xlabel('Feature index')
    ax7.set_ylabel('Value')
    ax7.grid(True, alpha=0.3)
    
    # 8. So sánh tỷ lệ Coherent/Incoherent
    ax8 = plt.subplot(3, 4, 8)
    total_coherent_bgr = ccv_bgr_coherent.sum()
    total_incoherent_bgr = ccv_bgr_incoherent.sum()
    total_coherent_hsv = ccv_hsv_coherent.sum()
    total_incoherent_hsv = ccv_hsv_incoherent.sum()
    
    categories = ['BGR', 'HSV']
    coherent_vals = [total_coherent_bgr, total_coherent_hsv]
    incoherent_vals = [total_incoherent_bgr, total_incoherent_hsv]
    
    x_cat = np.arange(len(categories))
    ax8.bar(x_cat - width/2, coherent_vals, width, label='Coherent', 
            alpha=0.8, color='green')
    ax8.bar(x_cat + width/2, incoherent_vals, width, label='Incoherent', 
            alpha=0.8, color='red')
    ax8.set_title('Tỷ lệ Coherent/Incoherent\n(BGR vs HSV)', 
                  fontsize=9, fontweight='bold')
    ax8.set_xticks(x_cat)
    ax8.set_xticklabels(categories)
    ax8.set_ylabel('Sum of values')
    ax8.legend(fontsize=8)
    ax8.grid(True, alpha=0.3, axis='y')
    
    # 9. So sánh kích thước vector với bins khác nhau
    ax9 = plt.subplot(3, 4, 9)
    feature_names = ['BGR\n(8 bins)', 'HSV\n(8 bins)', 'BGR\n(16 bins)']
    feature_sizes = [len(ccv_bgr_8), len(ccv_hsv_8), len(ccv_bgr_16)]
    colors_bar = ['purple', 'orange', 'blue']
    bars = ax9.bar(feature_names, feature_sizes, color=colors_bar, alpha=0.7)
    ax9.set_title('So sánh kích thước vector\nvới bins khác nhau', 
                  fontsize=9, fontweight='bold')
    ax9.set_ylabel('Số chiều (dimensions)')
    ax9.grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        ax9.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold')
    
    # 10. Phân bố Coherent (BGR)
    ax10 = plt.subplot(3, 4, 10)
    mask_coh = ccv_bgr_coherent > 0.001
    if mask_coh.sum() > 0:
        ax10.bar(x[mask_coh], ccv_bgr_coherent[mask_coh], 
                 alpha=0.8, color='green', width=1.0)
    ax10.set_title('BGR - Chỉ Coherent pixels\n(vùng màu lớn, liên tục)', 
                   fontsize=9, fontweight='bold')
    ax10.set_xlabel('Color bin')
    ax10.set_ylabel('Frequency')
    ax10.grid(True, alpha=0.3)
    
    # 11. Phân bố Incoherent (BGR)
    ax11 = plt.subplot(3, 4, 11)
    mask_incoh = ccv_bgr_incoherent > 0.001
    if mask_incoh.sum() > 0:
        ax11.bar(x[mask_incoh], ccv_bgr_incoherent[mask_incoh], 
                 alpha=0.8, color='red', width=1.0)
    ax11.set_title('BGR - Chỉ Incoherent pixels\n(vùng màu nhỏ, rời rạc)', 
                   fontsize=9, fontweight='bold')
    ax11.set_xlabel('Color bin')
    ax11.set_ylabel('Frequency')
    ax11.grid(True, alpha=0.3)
    
    # 12. CCV 16 bins
    ax12 = plt.subplot(3, 4, 12)
    ax12.bar(range(len(ccv_bgr_16)), ccv_bgr_16, alpha=0.7, color='blue', width=1.0)
    ax12.set_title(f'CCV BGR (16 bins)\nShape: {ccv_bgr_16.shape}\n(Chi tiết màu cao hơn)', 
                   fontsize=9, fontweight='bold')
    ax12.set_xlabel('Feature index')
    ax12.set_ylabel('Value')
    ax12.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ccv_feature_visualization.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Đã lưu visualization vào: ccv_feature_visualization.png")
    plt.show()
    
    # In thông tin chi tiết
    print("\n" + "="*80)
    print("THÔNG TIN CHI TIẾT VỀ ĐẶC TRƯNG CCV (COLOR COHERENCE VECTOR)")
    print("="*80)
    print(f"\n📸 Ảnh: {image_path}")
    print(f"📏 Kích thước ảnh: {image.shape[1]}×{image.shape[0]} pixels")
    print(f"📊 Tổng số pixel: {image.shape[0] * image.shape[1]:,}")
    
    print("\n" + "-"*80)
    print("VECTOR ĐẶC TRƯNG CCV")
    print("-"*80)
    
    print(f"\n1️⃣  CCV BGR (8 bins = 512 màu):")
    print(f"   • Số chiều vector: {len(ccv_bgr_8)} (512 màu × 2)")
    print(f"   • Shape: {ccv_bgr_8.shape}")
    print(f"   • Min value: {ccv_bgr_8.min():.6f}")
    print(f"   • Max value: {ccv_bgr_8.max():.6f}")
    print(f"   • Mean value: {ccv_bgr_8.mean():.6f}")
    print(f"   • Sum: {ccv_bgr_8.sum():.6f}")
    print(f"   • Total Coherent: {total_coherent_bgr:.6f}")
    print(f"   • Total Incoherent: {total_incoherent_bgr:.6f}")
    print(f"   • Coherent ratio: {total_coherent_bgr/(total_coherent_bgr+total_incoherent_bgr)*100:.2f}%")
    print(f"   • Sample values (first 10): {ccv_bgr_8[:10]}")
    
    print(f"\n2️⃣  CCV HSV (8 bins = 512 màu):")
    print(f"   • Số chiều vector: {len(ccv_hsv_8)} (512 màu × 2)")
    print(f"   • Shape: {ccv_hsv_8.shape}")
    print(f"   • Min value: {ccv_hsv_8.min():.6f}")
    print(f"   • Max value: {ccv_hsv_8.max():.6f}")
    print(f"   • Mean value: {ccv_hsv_8.mean():.6f}")
    print(f"   • Sum: {ccv_hsv_8.sum():.6f}")
    print(f"   • Total Coherent: {total_coherent_hsv:.6f}")
    print(f"   • Total Incoherent: {total_incoherent_hsv:.6f}")
    print(f"   • Coherent ratio: {total_coherent_hsv/(total_coherent_hsv+total_incoherent_hsv)*100:.2f}%")
    print(f"   • Sample values (first 10): {ccv_hsv_8[:10]}")
    
    print(f"\n3️⃣  CCV BGR (16 bins = 4096 màu):")
    print(f"   • Số chiều vector: {len(ccv_bgr_16)} (4096 màu × 2)")
    print(f"   • Shape: {ccv_bgr_16.shape}")
    print(f"   • Min value: {ccv_bgr_16.min():.6f}")
    print(f"   • Max value: {ccv_bgr_16.max():.6f}")
    print(f"   • Mean value: {ccv_bgr_16.mean():.6f}")
    print(f"   • Sum: {ccv_bgr_16.sum():.6f}")
    
    print("\n" + "="*80)
    print("GIẢI THÍCH")
    print("="*80)
    print("""
🔹 CCV (Color Coherence Vector) mở rộng histogram bằng cách phân loại pixel
   thành hai loại: Coherent (liên kết) và Incoherent (không liên kết).

🔹 Coherent pixels:
   • Thuộc vùng màu lớn và liên tục (≥ threshold pixel)
   • Phản ánh màu chính của đối tượng
   • Ví dụ: Nền đỏ của biển báo cấm

🔹 Incoherent pixels:
   • Thuộc vùng màu nhỏ và rời rạc (< threshold pixel)
   • Phản ánh nhiễu, chi tiết nhỏ, viền
   • Ví dụ: Nhiễu, shadow, reflection

🔹 Ưu điểm so với Histogram thông thường:
   • Mô tả cả phân bố màu VÀ cấu trúc không gian
   • Phân biệt được ảnh có cùng màu nhưng khác bố cục
   • Hiệu quả cho biển báo vì biển có màu sắc vùng lớn

🔹 Vector CCV:
   • Với n_bins=8: 8×8×8 = 512 màu → 512×2 = 1024 chiều
   • Với n_bins=16: 16×16×16 = 4096 màu → 4096×2 = 8192 chiều
   • Mỗi màu có 2 giá trị: [coherent_count, incoherent_count]
   • Đã chuẩn hóa: tổng = 1.0

🔹 Tham số quan trọng:
   • n_bins: Độ chi tiết màu (8-16 là phù hợp)
   • threshold: Ngưỡng phân loại coherent (100 pixels là tốt cho ảnh 128×128)
   • color_space: HSV tốt hơn BGR cho biển báo
    """)
    print("="*80)


def main():
    """Hàm main"""
    print("\n" + "="*80)
    print("DEMO: CHIẾT XUẤT ĐẶC TRƯNG CCV (COLOR COHERENCE VECTOR)")
    print("="*80)
    
    # Lấy ngẫu nhiên một ảnh từ train
    image_path, class_name = get_random_image_from_train()
    print(f"\n🎲 Ảnh được chọn ngẫu nhiên:")
    print(f"   • Class: {class_name}")
    print(f"   • File: {image_path}")
    
    # Visualize CCV features
    print(f"\n🔄 Đang chiết xuất và visualize đặc trưng CCV...")
    visualize_ccv_features(image_path)


if __name__ == "__main__":
    main()
