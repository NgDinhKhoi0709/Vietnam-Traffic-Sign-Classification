"""
Chiết xuất đặc trưng HOG (Histogram of Oriented Gradients)

Module này cung cấp các hàm để trích xuất đặc trưng HOG từ ảnh biển báo giao thông.
HOG là một phương pháp mô tả hình dạng và cấu trúc của đối tượng trong ảnh bằng cách
tính toán và lưu trữ phân bố của các hướng gradient trong ảnh.

Nguyên lý hoạt động:
1. Chia ảnh thành các ô (cells) nhỏ
2. Tính gradient (độ biến thiên cường độ) cho mỗi pixel
3. Tạo histogram của các hướng gradient trong mỗi ô
4. Chuẩn hóa các histogram trong các khối (blocks) lớn hơn
5. Kết hợp tất cả các histogram thành vector đặc trưng cuối cùng

Ưu điểm:
- Bất biến với các thay đổi về ánh sáng cục bộ
- Hiệu quả trong việc mô tả hình dạng và cấu trúc
- Phù hợp cho bài toán phân loại biển báo giao thông

Tham số quan trọng:
- orientations: Số lượng bin trong histogram (thường là 9)
- pixels_per_cell: Kích thước của mỗi ô (thường là 8x8 pixel)
- cells_per_block: Số ô trong mỗi khối để chuẩn hóa (thường là 2x2)
"""

import cv2
import numpy as np
from skimage.feature import hog
from skimage import exposure
import os
import pickle
import argparse
from datetime import datetime
from tqdm import tqdm


def extract_hog_features(image, orientations=9, pixels_per_cell=(8, 8), 
                        cells_per_block=(2, 2), visualize=False, multichannel=False):
    """
    Trích xuất đặc trưng HOG từ một ảnh
    
    Tham số:
    ----------
    image : numpy.ndarray
        Ảnh đầu vào (có thể là ảnh xám hoặc ảnh màu)
    orientations : int, mặc định=9
        Số lượng bin hướng gradient trong histogram (9 bin tương ứng với 20 độ mỗi bin)
    pixels_per_cell : tuple, mặc định=(8, 8)
        Kích thước của mỗi ô tính bằng pixel (chiều cao, chiều rộng)
        Mỗi ô sẽ có một histogram riêng
    cells_per_block : tuple, mặc định=(2, 2)
        Số lượng ô trong mỗi khối để chuẩn hóa (chiều cao, chiều rộng)
        Chuẩn hóa theo khối giúp bất biến với thay đổi ánh sáng
    visualize : bool, mặc định=False
        Nếu True, trả về cả ảnh trực quan hóa HOG
    multichannel : bool, mặc định=False
        Nếu True, xử lý ảnh màu đa kênh (BGR hoặc RGB)
        Nếu False, chuyển ảnh sang xám trước khi xử lý
    
    Trả về:
    ----------
    features : numpy.ndarray
        Vector đặc trưng HOG 1 chiều
        Kích thước phụ thuộc vào kích thước ảnh và các tham số
    hog_image : numpy.ndarray (tùy chọn)
        Ảnh trực quan hóa HOG (chỉ khi visualize=True)
    
    Ví dụ:
    ----------
    >>> import cv2
    >>> # Đọc ảnh biển báo
    >>> img = cv2.imread('vn-signs/train/Cam/Cam_1.jpg')
    >>> # Trích xuất đặc trưng HOG
    >>> features = extract_hog_features(img)
    >>> print(f"Kích thước vector đặc trưng: {features.shape}")
    >>> # Trích xuất và hiển thị HOG
    >>> features, hog_img = extract_hog_features(img, visualize=True)
    >>> cv2.imshow('HOG', hog_img)
    >>> cv2.waitKey(0)
    
    Ghi chú:
    ----------
    - Ảnh nên được resize về kích thước cố định trước khi trích xuất để đảm bảo
      vector đặc trưng có cùng kích thước cho tất cả ảnh
    - Với ảnh màu, nên chuyển sang ảnh xám (multichannel=False) để giảm độ phức tạp
    - Vector đặc trưng HOG thường rất dài, có thể cần giảm chiều sau đó
    """
    # Chuyển sang ảnh xám nếu cần
    if len(image.shape) == 3 and not multichannel:
        image_processed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_processed = image
    
    # Trích xuất đặc trưng HOG
    if visualize:
        features, hog_image = hog(
            image_processed,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            visualize=True,
            channel_axis=-1 if multichannel and len(image.shape) == 3 else None
        )
        
        # Chuẩn hóa ảnh HOG để hiển thị tốt hơn
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        
        return features, hog_image_rescaled
    else:
        features = hog(
            image_processed,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            visualize=False,
            channel_axis=-1 if multichannel and len(image.shape) == 3 else None
        )
        
        return features


def extract_hog_from_file(image_path, target_size=(128, 128), **kwargs):
    """
    Trích xuất đặc trưng HOG từ file ảnh
    
    Tham số:
    ----------
    image_path : str
        Đường dẫn đến file ảnh
    target_size : tuple, mặc định=(128, 128)
        Kích thước ảnh mục tiêu (chiều rộng, chiều cao) để resize
        Việc resize đảm bảo tất cả ảnh có cùng kích thước vector đặc trưng
    **kwargs : dict
        Các tham số bổ sung cho hàm extract_hog_features
        (orientations, pixels_per_cell, cells_per_block, v.v.)
    
    Trả về:
    ----------
    features : numpy.ndarray hoặc None
        Vector đặc trưng HOG, hoặc None nếu không đọc được ảnh
    
    Ví dụ:
    ----------
    >>> features = extract_hog_from_file('vn-signs/train/Cam/Cam_1.jpg')
    >>> if features is not None:
    >>>     print(f"Trích xuất thành công: {features.shape}")
    """
    # Đọc ảnh
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Không thể đọc ảnh: {image_path}")
        return None
    
    # Resize ảnh về kích thước cố định
    image_resized = cv2.resize(image, target_size)
    
    # Trích xuất đặc trưng
    features = extract_hog_features(image_resized, **kwargs)
    
    return features


def save_features_to_cache(cache_file, features_list, labels_list, class_names, target_size, **params):
    """
    Lưu đặc trưng đã trích xuất vào file cache
    
    Tham số:
    ----------
    cache_file : str
        Đường dẫn file để lưu cache
    features_list : list of numpy.ndarray
        Danh sách các vector đặc trưng
    labels_list : list of str
        Danh sách các nhãn
    class_names : list of str
        Danh sách tên các lớp
    target_size : tuple
        Kích thước ảnh đã sử dụng
    **params : dict
        Các tham số đã sử dụng khi trích xuất
    """
    # Tạo thư mục nếu chưa có
    cache_dir = os.path.dirname(cache_file)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    
    cache_data = {
        'features': features_list,
        'labels': labels_list,
        'class_names': class_names,
        'target_size': target_size,
        'params': params,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)
    
    print(f"💾 Đã lưu cache vào: {cache_file}")


def load_features_from_cache(cache_file):
    """
    Load đặc trưng từ file cache
    
    Tham số:
    ----------
    cache_file : str
        Đường dẫn file cache
    
    Trả về:
    ----------
    (features_list, labels_list, class_names) : tuple hoặc None
        Tuple chứa features, labels, class_names; hoặc None nếu không load được
    """
    if not os.path.exists(cache_file):
        return None
    
    try:
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        
        print(f"✅ Đã load cache từ: {cache_file}")
        print(f"   (Tạo lúc: {cache_data['timestamp']})")
        
        return cache_data['features'], cache_data['labels'], cache_data['class_names']
    except Exception as e:
        print(f"⚠️  Lỗi khi load cache: {e}")
        return None


def extract_hog_from_dataset(data_dir, target_size=(128, 128), use_cache=False, cache_file=None, **kwargs):
    """
    Trích xuất đặc trưng HOG từ toàn bộ dataset
    
    Hàm này duyệt qua tất cả các thư mục con trong data_dir, mỗi thư mục con
    là một lớp (class) của biển báo. Trích xuất đặc trưng HOG cho tất cả ảnh
    và trả về cùng với nhãn tương ứng.
    
    Tham số:
    ----------
    data_dir : str
        Đường dẫn đến thư mục chứa dataset
        Cấu trúc: data_dir/class_name/image_files
    target_size : tuple, mặc định=(128, 128)
        Kích thước ảnh mục tiêu để resize
    use_cache : bool, mặc định=False
        Có sử dụng cache không (load từ cache nếu có, lưu cache sau khi trích xuất)
    cache_file : str, mặc định=None
        Đường dẫn file cache. Nếu None, tự động tạo tên từ data_dir và tham số
    **kwargs : dict
        Các tham số bổ sung cho hàm extract_hog_features
        (orientations, pixels_per_cell, cells_per_block, v.v.)
    
    Trả về:
    ----------
    features_list : list of numpy.ndarray
        Danh sách các vector đặc trưng HOG
    labels_list : list of str
        Danh sách các nhãn tương ứng với mỗi vector đặc trưng
    class_names : list of str
        Danh sách các tên lớp trong dataset
    
    Ví dụ:
    ----------
    >>> # Không dùng cache
    >>> features, labels, classes = extract_hog_from_dataset('vn-signs/train')
    >>> 
    >>> # Sử dụng cache
    >>> features, labels, classes = extract_hog_from_dataset(
    >>>     'vn-signs/train',
    >>>     use_cache=True,
    >>>     cache_file='features_cache/hog_train.pkl'
    >>> )
    """
    # Tự động tạo tên file cache nếu không được cung cấp
    if use_cache and cache_file is None:
        # Tạo tên file từ data_dir và tham số
        dataset_name = os.path.basename(data_dir.rstrip('/\\'))
        params_str = f"{target_size[0]}x{target_size[1]}"
        params_str += f"_o{kwargs.get('orientations', 9)}"
        params_str += f"_ppc{kwargs.get('pixels_per_cell', (8,8))[0]}"
        params_str += f"_cpb{kwargs.get('cells_per_block', (2,2))[0]}"
        cache_file = f"features_cache/hog_{dataset_name}_{params_str}.pkl"
    
    # Thử load từ cache
    if use_cache and cache_file:
        cached_data = load_features_from_cache(cache_file)
        if cached_data is not None:
            return cached_data
        else:
            print("⚠️  Cache không tồn tại hoặc không hợp lệ, tiến hành trích xuất...")
    
    # Trích xuất đặc trưng
    features_list = []
    labels_list = []
    class_names = []
    
    # Lấy danh sách các lớp (thư mục con)
    if not os.path.exists(data_dir):
        print(f"Thư mục không tồn tại: {data_dir}")
        return features_list, labels_list, class_names
    
    class_dirs = [d for d in os.listdir(data_dir) 
                  if os.path.isdir(os.path.join(data_dir, d))]
    class_names = sorted(class_dirs)
    
    print(f"Tìm thấy {len(class_names)} lớp: {class_names}")
    
    # Duyệt qua từng lớp
    for class_name in class_names:
        class_path = os.path.join(data_dir, class_name)
        print(f"\nĐang xử lý lớp: {class_name}")
        
        # Lấy danh sách file ảnh
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"  Số lượng ảnh: {len(image_files)}")
        
        # Trích xuất đặc trưng cho từng ảnh
        for image_file in tqdm(image_files, desc=f"  {class_name}", leave=False):
            image_path = os.path.join(class_path, image_file)
            
            features = extract_hog_from_file(image_path, target_size, **kwargs)
            
            if features is not None:
                features_list.append(features)
                labels_list.append(class_name)
    
    print(f"\n=== Hoàn thành ===")
    print(f"Tổng số ảnh đã trích xuất: {len(features_list)}")
    
    # Lưu cache nếu được yêu cầu
    if use_cache and cache_file and len(features_list) > 0:
        save_features_to_cache(cache_file, features_list, labels_list, class_names, target_size, **kwargs)
    
    return features_list, labels_list, class_names


if __name__ == "__main__":
    # Thiết lập argument parser
    parser = argparse.ArgumentParser(
        description='Trích xuất đặc trưng HOG (Histogram of Oriented Gradients) từ dataset biển báo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--data-dir', type=str, default='vn-signs/train',
                        help='Đường dẫn đến thư mục dataset')
    parser.add_argument('--target-size', type=int, nargs=2, default=[128, 128],
                        metavar=('WIDTH', 'HEIGHT'),
                        help='Kích thước ảnh mục tiêu (width height)')
    parser.add_argument('--orientations', type=int, default=9,
                        help='Số lượng bin hướng gradient (9 → 20° mỗi bin)')
    parser.add_argument('--pixels-per-cell', type=int, nargs=2, default=[8, 8],
                        metavar=('H', 'W'),
                        help='Kích thước mỗi cell (height width)')
    parser.add_argument('--cells-per-block', type=int, nargs=2, default=[2, 2],
                        metavar=('H', 'W'),
                        help='Số cell trong mỗi block (height width)')
    parser.add_argument('--multichannel', action='store_true',
                        help='Xử lý ảnh màu đa kênh (mặc định: chuyển sang grayscale)')
    parser.add_argument('--use-cache', action='store_true', default=True,
                        help='Sử dụng cache để tăng tốc')
    parser.add_argument('--cache-file', type=str, default=None,
                        help='Đường dẫn file cache (tự động nếu không chỉ định)')
    
    args = parser.parse_args()
    
    # Chuyển đổi thành tuple
    target_size = tuple(args.target_size)
    pixels_per_cell = tuple(args.pixels_per_cell)
    cells_per_block = tuple(args.cells_per_block)
    data_dir = args.data_dir
    
    print("=" * 70)
    print("TRÍCH XUẤT ĐẶC TRƯNG HOG - HISTOGRAM OF ORIENTED GRADIENTS")
    print("=" * 70)
    print(f"\n📁 Dataset: {data_dir}")
    print(f"📐 Kích thước ảnh: {target_size}")
    print(f"🧭 Orientations: {args.orientations}")
    print(f"📦 Pixels per cell: {pixels_per_cell}")
    print(f"🔲 Cells per block: {cells_per_block}")
    print(f"🎨 Multichannel: {'BẬT' if args.multichannel else 'TẮT (grayscale)'}")
    print(f"💾 Cache: {'BẬT' if args.use_cache else 'TẮT'}")
    print()
    
    # Thống kê dataset
    print("=" * 70)
    print("THỐNG KÊ DATASET")
    print("=" * 70)
    
    if os.path.exists(data_dir):
        class_dirs = [d for d in os.listdir(data_dir) 
                     if os.path.isdir(os.path.join(data_dir, d))]
        print(f"\n✓ Số lớp: {len(class_dirs)}")
        print(f"✓ Tên các lớp: {sorted(class_dirs)}")
        
        # Đếm số ảnh trong mỗi lớp
        print(f"\n📊 Phân bố số ảnh theo lớp:")
        total_images = 0
        for class_name in sorted(class_dirs):
            class_path = os.path.join(data_dir, class_name)
            num_images = len([f for f in os.listdir(class_path) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            print(f"   - {class_name:12s}: {num_images:4d} ảnh")
            total_images += num_images
        print(f"\n✓ Tổng số ảnh: {total_images}")
    else:
        print(f"\n❌ Thư mục không tồn tại: {data_dir}")
        exit(1)
    
    # Trích xuất đặc trưng
    print("\n" + "=" * 70)
    print("TRÍCH XUẤT ĐẶC TRƯNG")
    print("=" * 70)
    
    features_list, labels_list, class_names = extract_hog_from_dataset(
        data_dir,
        target_size=target_size,
        orientations=args.orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        multichannel=args.multichannel,
        use_cache=args.use_cache,
        cache_file=args.cache_file
    )
    
    if len(features_list) > 0:
        print("\n" + "=" * 70)
        print("KẾT QUẢ")
        print("=" * 70)
        print(f"\n✓ Tổng số ảnh đã trích xuất: {len(features_list)}")
        print(f"✓ Số lớp: {len(class_names)}")
        print(f"✓ Kích thước mỗi vector: {features_list[0].shape}")
        print(f"✓ Tổng số chiều: {features_list[0].shape[0]}")
        
        print("\n" + "=" * 70)
        print("✅ HOÀN THÀNH!")
        print("=" * 70)
    else:
        print("\n❌ Không trích xuất được đặc trưng nào!")
