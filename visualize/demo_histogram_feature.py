"""
Demo: Chiết xuất và visualize đặc trưng Color Histogram từ một ảnh ngẫu nhiên
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from histogram_feature import extract_histogram_features


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


def visualize_histogram_features(image_path):
    """
    Chiết xuất và visualize đặc trưng color histogram từ một ảnh
    
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
    
    # Chiết xuất đặc trưng với các không gian màu khác nhau
    hist_bgr = extract_histogram_features(image, color_space='BGR', bins=(8, 8, 8))
    hist_hsv = extract_histogram_features(image, color_space='HSV', bins=(18, 8, 8))
    hist_gray = extract_histogram_features(image, color_space='GRAY', bins=(32,))
    
    # Tạo figure với nhiều subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Hiển thị ảnh gốc
    ax1 = plt.subplot(3, 3, 1)
    ax1.imshow(image_rgb)
    ax1.set_title(f'Ảnh gốc\n{os.path.basename(image_path)}\nKích thước: {image.shape[1]}x{image.shape[0]}', 
                  fontsize=10, fontweight='bold')
    ax1.axis('off')
    
    # 2. Histogram BGR (3D flattened)
    ax2 = plt.subplot(3, 3, 2)
    ax2.bar(range(len(hist_bgr)), hist_bgr, color='purple', alpha=0.7, width=1.0)
    ax2.set_title(f'BGR Histogram (8×8×8 = 512 bins)\nVector shape: {hist_bgr.shape}', 
                  fontsize=10, fontweight='bold')
    ax2.set_xlabel('Bin index')
    ax2.set_ylabel('Normalized frequency')
    ax2.grid(True, alpha=0.3)
    
    # 3. Histogram của từng kênh BGR riêng lẻ
    ax3 = plt.subplot(3, 3, 3)
    colors = ('b', 'g', 'r')
    labels = ('Blue', 'Green', 'Red')
    for i, (color, label) in enumerate(zip(colors, labels)):
        hist = cv2.calcHist([image], [i], None, [32], [0, 256])
        hist = hist / hist.sum()  # Normalize
        ax3.plot(hist, color=color, label=label, linewidth=2)
    ax3.set_title('Histogram từng kênh BGR (32 bins)', fontsize=10, fontweight='bold')
    ax3.set_xlabel('Bin (0-255)')
    ax3.set_ylabel('Normalized frequency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Hiển thị ảnh HSV
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ax4 = plt.subplot(3, 3, 4)
    ax4.imshow(image_hsv)
    ax4.set_title('Ảnh trong không gian HSV\n(H: Hue, S: Saturation, V: Value)', 
                  fontsize=10, fontweight='bold')
    ax4.axis('off')
    
    # 5. Histogram HSV (3D flattened)
    ax5 = plt.subplot(3, 3, 5)
    ax5.bar(range(len(hist_hsv)), hist_hsv, color='orange', alpha=0.7, width=1.0)
    ax5.set_title(f'HSV Histogram (18×8×8 = 1152 bins)\nVector shape: {hist_hsv.shape}', 
                  fontsize=10, fontweight='bold')
    ax5.set_xlabel('Bin index')
    ax5.set_ylabel('Normalized frequency')
    ax5.grid(True, alpha=0.3)
    
    # 6. Histogram của từng kênh HSV riêng lẻ
    ax6 = plt.subplot(3, 3, 6)
    # Hue (0-179)
    hist_h = cv2.calcHist([image_hsv], [0], None, [18], [0, 180])
    hist_h = hist_h / hist_h.sum()
    ax6.plot(hist_h, color='red', label='Hue (18 bins)', linewidth=2)
    ax6.set_title('HSV - Hue Channel', fontsize=10, fontweight='bold')
    ax6.set_xlabel('Hue bin (0-179)')
    ax6.set_ylabel('Normalized frequency')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Hiển thị ảnh grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ax7 = plt.subplot(3, 3, 7)
    ax7.imshow(image_gray, cmap='gray')
    ax7.set_title('Ảnh Grayscale', fontsize=10, fontweight='bold')
    ax7.axis('off')
    
    # 8. Histogram Grayscale
    ax8 = plt.subplot(3, 3, 8)
    ax8.bar(range(len(hist_gray)), hist_gray, color='gray', alpha=0.7, width=1.0)
    ax8.set_title(f'Grayscale Histogram (32 bins)\nVector shape: {hist_gray.shape}', 
                  fontsize=10, fontweight='bold')
    ax8.set_xlabel('Bin index')
    ax8.set_ylabel('Normalized frequency')
    ax8.grid(True, alpha=0.3)
    
    # 9. So sánh kích thước vector đặc trưng
    ax9 = plt.subplot(3, 3, 9)
    feature_names = ['BGR\n(8×8×8)', 'HSV\n(18×8×8)', 'GRAY\n(32)']
    feature_sizes = [len(hist_bgr), len(hist_hsv), len(hist_gray)]
    colors_bar = ['purple', 'orange', 'gray']
    bars = ax9.bar(feature_names, feature_sizes, color=colors_bar, alpha=0.7)
    ax9.set_title('So sánh kích thước vector đặc trưng', fontsize=10, fontweight='bold')
    ax9.set_ylabel('Số chiều (dimensions)')
    ax9.grid(True, alpha=0.3, axis='y')
    
    # Thêm giá trị lên các cột
    for bar in bars:
        height = bar.get_height()
        ax9.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('histogram_feature_visualization.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Đã lưu visualization vào: histogram_feature_visualization.png")
    plt.show()
    
    # In thông tin chi tiết
    print("\n" + "="*80)
    print("THÔNG TIN CHI TIẾT VỀ ĐẶC TRƯNG COLOR HISTOGRAM")
    print("="*80)
    print(f"\n📸 Ảnh: {image_path}")
    print(f"📏 Kích thước ảnh: {image.shape[1]}×{image.shape[0]} pixels")
    print(f"🎨 Số kênh màu: {image.shape[2] if len(image.shape) == 3 else 1}")
    
    print("\n" + "-"*80)
    print("VECTOR ĐẶC TRƯNG")
    print("-"*80)
    
    print(f"\n1️⃣  BGR Histogram (8×8×8 bins):")
    print(f"   • Số chiều: {len(hist_bgr)}")
    print(f"   • Shape: {hist_bgr.shape}")
    print(f"   • Min value: {hist_bgr.min():.6f}")
    print(f"   • Max value: {hist_bgr.max():.6f}")
    print(f"   • Mean value: {hist_bgr.mean():.6f}")
    print(f"   • Sum: {hist_bgr.sum():.6f}")
    print(f"   • Sample values (first 10): {hist_bgr[:10]}")
    
    print(f"\n2️⃣  HSV Histogram (18×8×8 bins):")
    print(f"   • Số chiều: {len(hist_hsv)}")
    print(f"   • Shape: {hist_hsv.shape}")
    print(f"   • Min value: {hist_hsv.min():.6f}")
    print(f"   • Max value: {hist_hsv.max():.6f}")
    print(f"   • Mean value: {hist_hsv.mean():.6f}")
    print(f"   • Sum: {hist_hsv.sum():.6f}")
    print(f"   • Sample values (first 10): {hist_hsv[:10]}")
    
    print(f"\n3️⃣  Grayscale Histogram (32 bins):")
    print(f"   • Số chiều: {len(hist_gray)}")
    print(f"   • Shape: {hist_gray.shape}")
    print(f"   • Min value: {hist_gray.min():.6f}")
    print(f"   • Max value: {hist_gray.max():.6f}")
    print(f"   • Mean value: {hist_gray.mean():.6f}")
    print(f"   • Sum: {hist_gray.sum():.6f}")
    print(f"   • Sample values (all 32): {hist_gray}")
    
    print("\n" + "="*80)
    print("GIẢI THÍCH")
    print("="*80)
    print("""
🔹 Color Histogram mô tả phân bố màu sắc trong ảnh bằng cách đếm số pixel 
   trong mỗi khoảng màu (bin).

🔹 BGR Histogram (512 dims):
   • Chia mỗi kênh B, G, R thành 8 bins → 8×8×8 = 512 bins tổng cộng
   • Vector 512 chiều mô tả phân bố màu trong không gian BGR
   • Đơn giản nhưng nhạy cảm với thay đổi ánh sáng

🔹 HSV Histogram (1152 dims):
   • Hue: 18 bins (màu sắc độc lập với ánh sáng)
   • Saturation: 8 bins (độ bão hòa)
   • Value: 8 bins (độ sáng)
   • Tốt hơn BGR vì tách riêng màu và ánh sáng

🔹 Grayscale Histogram (32 dims):
   • Chỉ quan tâm đến cường độ sáng
   • Vector ngắn nhất, nhanh nhất
   • Mất thông tin về màu sắc

🔹 Các giá trị đã được chuẩn hóa (normalize=True):
   • Tổng các giá trị = 1.0
   • Bất biến với kích thước ảnh
   • Phù hợp để so sánh giữa các ảnh khác nhau
    """)
    print("="*80)


def main():
    """Hàm main"""
    print("\n" + "="*80)
    print("DEMO: CHIẾT XUẤT ĐẶC TRƯNG COLOR HISTOGRAM")
    print("="*80)
    
    # Lấy ngẫu nhiên một ảnh từ train
    image_path, class_name = get_random_image_from_train()
    print(f"\n🎲 Ảnh được chọn ngẫu nhiên:")
    print(f"   • Class: {class_name}")
    print(f"   • File: {image_path}")
    
    # Visualize histogram features
    print(f"\n🔄 Đang chiết xuất và visualize đặc trưng...")
    visualize_histogram_features(image_path)


if __name__ == "__main__":
    main()
