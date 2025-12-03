"""
Demo: Chiết xuất và visualize đặc trưng HOG (Histogram of Oriented Gradients) từ một ảnh ngẫu nhiên
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from hog_feature import extract_hog_features


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


def compute_gradient_magnitude_and_direction(image):
    """Tính gradient magnitude và direction"""
    # Chuyển sang grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Tính gradient theo x và y
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=1)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=1)
    
    # Tính magnitude và direction
    magnitude = np.sqrt(gx**2 + gy**2)
    direction = np.arctan2(gy, gx) * (180 / np.pi) % 180  # 0-180 độ
    
    return magnitude, direction, gx, gy


def visualize_hog_features(image_path):
    """
    Chiết xuất và visualize đặc trưng HOG từ một ảnh
    
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
    
    # Resize để có kích thước chuẩn
    target_size = (128, 128)
    image_resized = cv2.resize(image, target_size)
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    
    # Chiết xuất đặc trưng HOG với các cấu hình khác nhau
    # Cấu hình 1: 9 orientations, 8x8 pixels_per_cell, 2x2 cells_per_block
    hog_feat_1, hog_img_1 = extract_hog_features(
        image_resized, orientations=9, pixels_per_cell=(8, 8),
        cells_per_block=(2, 2), visualize=True
    )
    
    # Cấu hình 2: 12 orientations
    hog_feat_2, hog_img_2 = extract_hog_features(
        image_resized, orientations=12, pixels_per_cell=(8, 8),
        cells_per_block=(2, 2), visualize=True
    )
    
    # Cấu hình 3: 16x16 pixels_per_cell (ít chi tiết hơn)
    hog_feat_3, hog_img_3 = extract_hog_features(
        image_resized, orientations=9, pixels_per_cell=(16, 16),
        cells_per_block=(2, 2), visualize=True
    )
    
    # Tính gradient magnitude và direction
    magnitude, direction, gx, gy = compute_gradient_magnitude_and_direction(image_resized)
    
    # Tạo figure với nhiều subplots
    fig = plt.figure(figsize=(18, 14))
    
    # 1. Hiển thị ảnh gốc
    ax1 = plt.subplot(4, 4, 1)
    ax1.imshow(image_rgb)
    ax1.set_title(f'Ảnh gốc\n{os.path.basename(image_path)}\nResized: {target_size[0]}×{target_size[1]}', 
                  fontsize=9, fontweight='bold')
    ax1.axis('off')
    
    # 2. Ảnh grayscale
    ax2 = plt.subplot(4, 4, 2)
    ax2.imshow(image_gray, cmap='gray')
    ax2.set_title('Grayscale\n(HOG chỉ dùng grayscale)', fontsize=9, fontweight='bold')
    ax2.axis('off')
    
    # 3. Gradient X (Sobel)
    ax3 = plt.subplot(4, 4, 3)
    im3 = ax3.imshow(gx, cmap='RdBu')
    ax3.set_title('Gradient X (Sobel)\n(Cạnh dọc)', fontsize=9, fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046)
    
    # 4. Gradient Y (Sobel)
    ax4 = plt.subplot(4, 4, 4)
    im4 = ax4.imshow(gy, cmap='RdBu')
    ax4.set_title('Gradient Y (Sobel)\n(Cạnh ngang)', fontsize=9, fontweight='bold')
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046)
    
    # 5. Gradient Magnitude
    ax5 = plt.subplot(4, 4, 5)
    im5 = ax5.imshow(magnitude, cmap='hot')
    ax5.set_title('Gradient Magnitude\n(Độ mạnh cạnh)', fontsize=9, fontweight='bold')
    ax5.axis('off')
    plt.colorbar(im5, ax=ax5, fraction=0.046)
    
    # 6. Gradient Direction
    ax6 = plt.subplot(4, 4, 6)
    im6 = ax6.imshow(direction, cmap='hsv')
    ax6.set_title('Gradient Direction\n(Hướng cạnh: 0-180°)', fontsize=9, fontweight='bold')
    ax6.axis('off')
    plt.colorbar(im6, ax=ax6, fraction=0.046)
    
    # 7. HOG Visualization (9 orientations)
    ax7 = plt.subplot(4, 4, 7)
    ax7.imshow(hog_img_1, cmap='gray')
    ax7.set_title('HOG Visualization\n(9 orientations, 8×8 cell)', fontsize=9, fontweight='bold')
    ax7.axis('off')
    
    # 8. Overlay HOG trên ảnh gốc
    ax8 = plt.subplot(4, 4, 8)
    ax8.imshow(image_rgb, alpha=0.7)
    ax8.imshow(hog_img_1, cmap='gray', alpha=0.5)
    ax8.set_title('HOG Overlay\n(HOG + Original)', fontsize=9, fontweight='bold')
    ax8.axis('off')
    
    # 9. HOG Feature Vector (9 orientations)
    ax9 = plt.subplot(4, 4, 9)
    ax9.plot(hog_feat_1, linewidth=0.5, alpha=0.7, color='blue')
    ax9.set_title(f'HOG Vector (9 ori, 8×8 cell)\nShape: {hog_feat_1.shape}', 
                  fontsize=9, fontweight='bold')
    ax9.set_xlabel('Feature index')
    ax9.set_ylabel('Value')
    ax9.grid(True, alpha=0.3)
    
    # 10. Histogram của HOG features (9 ori)
    ax10 = plt.subplot(4, 4, 10)
    ax10.hist(hog_feat_1, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax10.set_title('Phân bố giá trị HOG\n(9 orientations)', fontsize=9, fontweight='bold')
    ax10.set_xlabel('Feature value')
    ax10.set_ylabel('Frequency')
    ax10.grid(True, alpha=0.3)
    
    # 11. HOG với 12 orientations
    ax11 = plt.subplot(4, 4, 11)
    ax11.imshow(hog_img_2, cmap='gray')
    ax11.set_title('HOG (12 orientations)\nChi tiết hơn về hướng', fontsize=9, fontweight='bold')
    ax11.axis('off')
    
    # 12. HOG Feature Vector (12 orientations)
    ax12 = plt.subplot(4, 4, 12)
    ax12.plot(hog_feat_2, linewidth=0.5, alpha=0.7, color='green')
    ax12.set_title(f'HOG Vector (12 ori, 8×8 cell)\nShape: {hog_feat_2.shape}', 
                   fontsize=9, fontweight='bold')
    ax12.set_xlabel('Feature index')
    ax12.set_ylabel('Value')
    ax12.grid(True, alpha=0.3)
    
    # 13. HOG với 16x16 pixels_per_cell
    ax13 = plt.subplot(4, 4, 13)
    ax13.imshow(hog_img_3, cmap='gray')
    ax13.set_title('HOG (16×16 cell)\nÍt chi tiết, vector ngắn hơn', fontsize=9, fontweight='bold')
    ax13.axis('off')
    
    # 14. HOG Feature Vector (16x16 cell)
    ax14 = plt.subplot(4, 4, 14)
    ax14.plot(hog_feat_3, linewidth=0.5, alpha=0.7, color='red')
    ax14.set_title(f'HOG Vector (9 ori, 16×16 cell)\nShape: {hog_feat_3.shape}', 
                   fontsize=9, fontweight='bold')
    ax14.set_xlabel('Feature index')
    ax14.set_ylabel('Value')
    ax14.grid(True, alpha=0.3)
    
    # 15. So sánh kích thước vector
    ax15 = plt.subplot(4, 4, 15)
    configs = ['9 ori\n8×8 cell', '12 ori\n8×8 cell', '9 ori\n16×16 cell']
    sizes = [len(hog_feat_1), len(hog_feat_2), len(hog_feat_3)]
    colors_bar = ['blue', 'green', 'red']
    bars = ax15.bar(configs, sizes, color=colors_bar, alpha=0.7)
    ax15.set_title('So sánh kích thước vector\nvới cấu hình khác nhau', 
                   fontsize=9, fontweight='bold')
    ax15.set_ylabel('Số chiều (dimensions)')
    ax15.grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        ax15.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height)}',
                 ha='center', va='bottom', fontweight='bold', fontsize=8)
    
    # 16. Distribution statistics
    ax16 = plt.subplot(4, 4, 16)
    stats_text = f"""HOG Statistics (9 ori, 8×8):
    
Min: {hog_feat_1.min():.4f}
Max: {hog_feat_1.max():.4f}
Mean: {hog_feat_1.mean():.4f}
Std: {hog_feat_1.std():.4f}
Median: {np.median(hog_feat_1):.4f}

Non-zero: {np.count_nonzero(hog_feat_1)}
({np.count_nonzero(hog_feat_1)/len(hog_feat_1)*100:.1f}%)
    """
    ax16.text(0.1, 0.5, stats_text, fontsize=9, family='monospace',
              verticalalignment='center')
    ax16.set_title('Thống kê HOG Features', fontsize=9, fontweight='bold')
    ax16.axis('off')
    
    plt.tight_layout()
    plt.savefig('hog_feature_visualization.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Đã lưu visualization vào: hog_feature_visualization.png")
    plt.show()
    
    # In thông tin chi tiết
    print("\n" + "="*80)
    print("THÔNG TIN CHI TIẾT VỀ ĐẶC TRƯNG HOG (HISTOGRAM OF ORIENTED GRADIENTS)")
    print("="*80)
    print(f"\n📸 Ảnh: {image_path}")
    print(f"📏 Kích thước gốc: {image.shape[1]}×{image.shape[0]} pixels")
    print(f"📏 Kích thước sau resize: {target_size[0]}×{target_size[1]} pixels")
    
    print("\n" + "-"*80)
    print("GRADIENT INFORMATION")
    print("-"*80)
    print(f"\n🔍 Gradient Magnitude:")
    print(f"   • Min: {magnitude.min():.4f}")
    print(f"   • Max: {magnitude.max():.4f}")
    print(f"   • Mean: {magnitude.mean():.4f}")
    print(f"   • Std: {magnitude.std():.4f}")
    
    print(f"\n🧭 Gradient Direction:")
    print(f"   • Range: 0-180 degrees")
    print(f"   • Mean: {direction.mean():.2f}°")
    print(f"   • Std: {direction.std():.2f}°")
    
    print("\n" + "-"*80)
    print("HOG FEATURE VECTORS")
    print("-"*80)
    
    print(f"\n1️⃣  HOG (9 orientations, 8×8 pixels_per_cell, 2×2 cells_per_block):")
    print(f"   • Số chiều vector: {len(hog_feat_1)}")
    print(f"   • Shape: {hog_feat_1.shape}")
    print(f"   • Min value: {hog_feat_1.min():.6f}")
    print(f"   • Max value: {hog_feat_1.max():.6f}")
    print(f"   • Mean value: {hog_feat_1.mean():.6f}")
    print(f"   • Std value: {hog_feat_1.std():.6f}")
    print(f"   • Non-zero elements: {np.count_nonzero(hog_feat_1)} ({np.count_nonzero(hog_feat_1)/len(hog_feat_1)*100:.2f}%)")
    print(f"   • Sample values (first 20): {hog_feat_1[:20]}")
    
    # Tính toán thông tin về cells và blocks
    cells_x = target_size[0] // 8  # 128 / 8 = 16
    cells_y = target_size[1] // 8  # 128 / 8 = 16
    blocks_x = cells_x - 2 + 1  # 16 - 2 + 1 = 15
    blocks_y = cells_y - 2 + 1  # 16 - 2 + 1 = 15
    features_per_block = 9 * 2 * 2  # orientations * cells_per_block
    total_features = blocks_x * blocks_y * features_per_block
    
    print(f"\n   📐 Cấu trúc:")
    print(f"      • Image: {target_size[0]}×{target_size[1]} pixels")
    print(f"      • Cells: {cells_x}×{cells_y} = {cells_x*cells_y} cells")
    print(f"      • Each cell: 8×8 pixels")
    print(f"      • Blocks: {blocks_x}×{blocks_y} = {blocks_x*blocks_y} blocks")
    print(f"      • Each block: 2×2 cells = 4 cells")
    print(f"      • Features per block: 9 ori × 4 cells = {features_per_block}")
    print(f"      • Total features: {blocks_x}×{blocks_y}×{features_per_block} = {total_features}")
    
    print(f"\n2️⃣  HOG (12 orientations, 8×8 pixels_per_cell, 2×2 cells_per_block):")
    print(f"   • Số chiều vector: {len(hog_feat_2)}")
    print(f"   • Shape: {hog_feat_2.shape}")
    print(f"   • Min value: {hog_feat_2.min():.6f}")
    print(f"   • Max value: {hog_feat_2.max():.6f}")
    print(f"   • Mean value: {hog_feat_2.mean():.6f}")
    print(f"   • Chi tiết hơn về hướng gradient (12 bins thay vì 9)")
    
    print(f"\n3️⃣  HOG (9 orientations, 16×16 pixels_per_cell, 2×2 cells_per_block):")
    print(f"   • Số chiều vector: {len(hog_feat_3)}")
    print(f"   • Shape: {hog_feat_3.shape}")
    print(f"   • Min value: {hog_feat_3.min():.6f}")
    print(f"   • Max value: {hog_feat_3.max():.6f}")
    print(f"   • Mean value: {hog_feat_3.mean():.6f}")
    print(f"   • Vector ngắn hơn vì cell lớn hơn (ít chi tiết hơn)")
    
    print("\n" + "="*80)
    print("GIẢI THÍCH")
    print("="*80)
    print("""
🔹 HOG (Histogram of Oriented Gradients) mô tả hình dạng và cấu trúc của đối tượng
   bằng cách tính phân bố hướng gradient (cạnh) trong ảnh.

🔹 Quy trình chiết xuất HOG:
   1. Chuyển ảnh sang grayscale
   2. Tính gradient (Gx, Gy) bằng Sobel filter
   3. Tính magnitude và direction của gradient
   4. Chia ảnh thành cells (8×8 pixels)
   5. Tạo histogram orientations cho mỗi cell (9 bins = 20° mỗi bin)
   6. Nhóm cells thành blocks (2×2 cells) và chuẩn hóa
   7. Kết hợp tất cả histogram thành vector đặc trưng

🔹 Ưu điểm của HOG:
   • Bất biến với thay đổi ánh sáng (do chuẩn hóa block)
   • Mô tả tốt hình dạng và cấu trúc (cạnh, góc, đường viền)
   • Hiệu quả cho phân loại đối tượng có hình dạng đặc trưng
   • Phù hợp với biển báo giao thông (hình tròn, tam giác, vuông)

🔹 Tham số quan trọng:
   • orientations (9): Số lượng bin hướng (9 → 180°/9 = 20° mỗi bin)
   • pixels_per_cell (8×8): Kích thước cell (càng nhỏ càng chi tiết)
   • cells_per_block (2×2): Số cell trong block để chuẩn hóa

🔹 Gradient magnitude: Độ mạnh của cạnh (thay đổi cường độ)
🔹 Gradient direction: Hướng của cạnh (0-180°)

🔹 Kích thước vector:
   • Phụ thuộc vào kích thước ảnh và tham số
   • Image 128×128, cell 8×8, block 2×2, 9 ori → {total_features} features
   • Vector càng dài càng chi tiết nhưng tốn bộ nhớ và có thể overfit
    """)
    print("="*80)


def main():
    """Hàm main"""
    print("\n" + "="*80)
    print("DEMO: CHIẾT XUẤT ĐẶC TRƯNG HOG (HISTOGRAM OF ORIENTED GRADIENTS)")
    print("="*80)
    
    # Lấy ngẫu nhiên một ảnh từ train
    image_path, class_name = get_random_image_from_train()
    print(f"\n🎲 Ảnh được chọn ngẫu nhiên:")
    print(f"   • Class: {class_name}")
    print(f"   • File: {image_path}")
    
    # Visualize HOG features
    print(f"\n🔄 Đang chiết xuất và visualize đặc trưng HOG...")
    visualize_hog_features(image_path)


if __name__ == "__main__":
    main()
