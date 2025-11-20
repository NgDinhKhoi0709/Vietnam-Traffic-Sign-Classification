"""
Chiết xuất đặc trưng Histogram (Histogram của màu sắc)

Module này cung cấp các hàm để trích xuất đặc trưng histogram màu từ ảnh biển báo giao thông.
Color Histogram là một phương pháp mô tả phân bố màu sắc trong ảnh bằng cách đếm số lượng
pixel cho mỗi khoảng màu (bin).

Nguyên lý hoạt động:
1. Chọn không gian màu phù hợp (BGR, RGB, HSV, Lab, v.v.)
2. Chia mỗi kênh màu thành các bin (khoảng giá trị)
3. Đếm số lượng pixel rơi vào mỗi bin
4. Kết hợp histogram của các kênh thành vector đặc trưng
5. Chuẩn hóa vector để bất biến với kích thước ảnh

Các không gian màu:
- BGR/RGB: Không gian màu cơ bản, đơn giản nhưng nhạy cảm với ánh sáng
- HSV: Tách màu sắc (Hue), độ bão hòa (Saturation), độ sáng (Value)
       Tốt cho phân tích màu độc lập với ánh sáng
- Lab: Perceptually uniform, tốt cho so sánh màu
- YCrCb: Tách độ sáng và màu, tốt cho xử lý ảnh

Ưu điểm:
- Đơn giản, nhanh, hiệu quả
- Bất biến với thay đổi vị trí, xoay, tỷ lệ
- Hiệu quả với ảnh có màu sắc đặc trưng (như biển báo)

Nhược điểm:
- Mất thông tin về vị trí không gian của màu
- Nhạy cảm với thay đổi ánh sáng (với BGR/RGB)
"""

import cv2
import numpy as np
import os
import pickle
import argparse
from datetime import datetime
from tqdm import tqdm


def extract_histogram_features(image, color_space='BGR', bins=(8, 8, 8), 
                               ranges=None, normalize=True):
    """
    Trích xuất đặc trưng histogram màu từ một ảnh
    
    Tham số:
    ----------
    image : numpy.ndarray
        Ảnh đầu vào (BGR format từ OpenCV)
    color_space : str, mặc định='BGR'
        Không gian màu để tính histogram
        Các lựa chọn: 'BGR', 'RGB', 'HSV', 'Lab', 'YCrCb', 'GRAY'
    bins : tuple of int, mặc định=(8, 8, 8)
        Số lượng bin cho mỗi kênh (channel_1, channel_2, channel_3)
        Ví dụ: (8, 8, 8) → 8x8x8 = 512 bins tổng cộng
        Với GRAY: chỉ cần một giá trị, ví dụ: bins=(32,)
    ranges : list of tuple, mặc định=None
        Khoảng giá trị cho mỗi kênh [(min1, max1), (min2, max2), (min3, max3)]
        Nếu None, sử dụng khoảng mặc định:
        - BGR, RGB: [0, 256] cho mỗi kênh
        - HSV: [(0, 180), (0, 256), (0, 256)] (H: 0-179, S,V: 0-255)
        - Lab: [(0, 256), (0, 256), (0, 256)]
        - GRAY: [(0, 256)]
    normalize : bool, mặc định=True
        Có chuẩn hóa histogram thành [0, 1] không
        Chuẩn hóa giúp bất biến với kích thước ảnh
    
    Trả về:
    ----------
    histogram : numpy.ndarray
        Vector histogram 1 chiều
        Kích thước = bins[0] * bins[1] * bins[2] (hoặc bins[0] với GRAY)
    
    Ví dụ:
    ----------
    >>> import cv2
    >>> # Đọc ảnh biển báo
    >>> img = cv2.imread('vn-signs/train/Cam/Cam_1.jpg')
    >>> 
    >>> # Histogram BGR cơ bản
    >>> hist_bgr = extract_histogram_features(img, color_space='BGR', bins=(8, 8, 8))
    >>> print(f"BGR histogram: {hist_bgr.shape}")  # (512,)
    >>> 
    >>> # Histogram HSV (tốt hơn với màu sắc)
    >>> hist_hsv = extract_histogram_features(img, color_space='HSV', bins=(18, 8, 8))
    >>> print(f"HSV histogram: {hist_hsv.shape}")  # (1152,)
    >>> 
    >>> # Histogram xám (đơn giản nhất)
    >>> hist_gray = extract_histogram_features(img, color_space='GRAY', bins=(32,))
    >>> print(f"GRAY histogram: {hist_gray.shape}")  # (32,)
    
    Ghi chú:
    ----------
    - HSV thường cho kết quả tốt nhất với biển báo vì bất biến hơn với ánh sáng
    - Hue (H) có khoảng [0, 179] trong OpenCV (không phải [0, 360])
    - Số bins càng lớn càng chi tiết nhưng vector càng dài và dễ overfit
    - Nên chuẩn hóa (normalize=True) để so sánh giữa các ảnh khác kích thước
    """
    # Chuyển đổi không gian màu
    if color_space == 'BGR':
        image_converted = image
        default_ranges = [(0, 256), (0, 256), (0, 256)]
    elif color_space == 'RGB':
        image_converted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        default_ranges = [(0, 256), (0, 256), (0, 256)]
    elif color_space == 'HSV':
        image_converted = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        default_ranges = [(0, 180), (0, 256), (0, 256)]  # H: 0-179, S,V: 0-255
    elif color_space == 'Lab':
        image_converted = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        default_ranges = [(0, 256), (0, 256), (0, 256)]
    elif color_space == 'YCrCb':
        image_converted = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        default_ranges = [(0, 256), (0, 256), (0, 256)]
    elif color_space == 'GRAY':
        image_converted = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        default_ranges = [(0, 256)]
        # Đảm bảo bins là tuple với 1 phần tử
        if isinstance(bins, int):
            bins = (bins,)
    else:
        raise ValueError(f"Không gian màu không hợp lệ: {color_space}")
    
    # Sử dụng ranges mặc định nếu không được cung cấp
    if ranges is None:
        ranges = default_ranges
    
    # Tính histogram
    if color_space == 'GRAY':
        # Histogram 1D cho ảnh xám
        histogram = cv2.calcHist([image_converted], [0], None, [bins[0]], ranges[0])
    else:
        # Histogram 3D cho ảnh màu
        histogram = cv2.calcHist(
            [image_converted], 
            [0, 1, 2],  # Tính cho cả 3 kênh
            None, 
            bins, 
            ranges[0] + ranges[1] + ranges[2]  # Flatten ranges
        )
    
    # Flatten thành vector 1D
    histogram = histogram.flatten()
    
    # Chuẩn hóa
    if normalize:
        histogram = histogram / (histogram.sum() + 1e-7)  # Tránh chia cho 0
    
    return histogram


def extract_histogram_from_file(image_path, target_size=(128, 128), **kwargs):
    """
    Trích xuất đặc trưng histogram từ file ảnh
    
    Tham số:
    ----------
    image_path : str
        Đường dẫn đến file ảnh
    target_size : tuple, mặc định=(128, 128)
        Kích thước ảnh mục tiêu (chiều rộng, chiều cao) để resize
    **kwargs : dict
        Các tham số bổ sung cho hàm extract_histogram_features
        (color_space, bins, normalize, v.v.)
    
    Trả về:
    ----------
    histogram : numpy.ndarray hoặc None
        Vector histogram, hoặc None nếu không đọc được ảnh
    
    Ví dụ:
    ----------
    >>> # Histogram HSV
    >>> hist = extract_histogram_from_file(
    >>>     'vn-signs/train/Cam/Cam_1.jpg',
    >>>     color_space='HSV',
    >>>     bins=(18, 8, 8)
    >>> )
    >>> 
    >>> # Histogram RGB
    >>> hist = extract_histogram_from_file(
    >>>     'vn-signs/train/Cam/Cam_1.jpg',
    >>>     color_space='RGB',
    >>>     bins=(8, 8, 8)
    >>> )
    """
    # Đọc ảnh
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Không thể đọc ảnh: {image_path}")
        return None
    
    # Resize ảnh
    image_resized = cv2.resize(image, target_size)
    
    # Trích xuất histogram
    histogram = extract_histogram_features(image_resized, **kwargs)
    
    return histogram


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


def extract_histogram_from_dataset(data_dir, target_size=(128, 128), use_cache=False, cache_file=None, **kwargs):
    """
    Trích xuất đặc trưng histogram từ toàn bộ dataset
    
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
        Các tham số bổ sung cho hàm extract_histogram_from_file
        (color_space, bins, normalize)
    
    Trả về:
    ----------
    features_list : list of numpy.ndarray
        Danh sách các vector histogram
    labels_list : list of str
        Danh sách các nhãn tương ứng
    class_names : list of str
        Danh sách các tên lớp trong dataset
    
    Ví dụ:
    ----------
    >>> # Không dùng cache
    >>> features, labels, classes = extract_histogram_from_dataset(
    >>>     'vn-signs/train',
    >>>     color_space='HSV',
    >>>     bins=(18, 8, 8)
    >>> )
    >>> 
    >>> # Sử dụng cache (khuyến nghị)
    >>> features, labels, classes = extract_histogram_from_dataset(
    >>>     'vn-signs/train',
    >>>     use_cache=True,
    >>>     color_space='HSV',
    >>>     bins=(18, 8, 8)
    >>> )
    """
    # Tự động tạo tên file cache nếu không được cung cấp
    if use_cache and cache_file is None:
        # Tạo tên file từ data_dir và tham số
        dataset_name = os.path.basename(data_dir.rstrip('/\\'))
        params_str = f"{target_size[0]}x{target_size[1]}"
        params_str += f"_{kwargs.get('color_space', 'HSV')}"
        bins = kwargs.get('bins', (18, 8, 8))
        if isinstance(bins, tuple):
            params_str += f"_bins{'x'.join(map(str, bins))}"
        else:
            params_str += f"_bins{bins}"
        cache_file = f"features_cache/histogram_{dataset_name}_{params_str}.pkl"
    
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
    
    # Lấy danh sách các lớp
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
            
            histogram = extract_histogram_from_file(image_path, target_size, **kwargs)
            
            if histogram is not None:
                features_list.append(histogram)
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
        description='Trích xuất đặc trưng Histogram màu sắc từ dataset biển báo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--data-dir', type=str, default='vn-signs/train',
                        help='Đường dẫn đến thư mục dataset')
    parser.add_argument('--target-size', type=int, nargs=2, default=[128, 128],
                        metavar=('WIDTH', 'HEIGHT'),
                        help='Kích thước ảnh mục tiêu (width height)')
    parser.add_argument('--color-space', type=str, default='HSV',
                        choices=['BGR', 'RGB', 'HSV', 'Lab', 'YCrCb', 'GRAY'],
                        help='Không gian màu (HSV khuyến nghị cho biển báo)')
    parser.add_argument('--bins', type=int, nargs='+', default=[18, 8, 8],
                        metavar='B',
                        help='Số bins cho mỗi kênh (18 8 8 cho HSV, 1 số cho GRAY)')
    parser.add_argument('--normalize', action='store_true', default=True,
                        help='Chuẩn hóa histogram')
    parser.add_argument('--use-cache', action='store_true', default=True,
                        help='Sử dụng cache để tăng tốc')
    parser.add_argument('--cache-file', type=str, default=None,
                        help='Đường dẫn file cache (tự động nếu không chỉ định)')
    
    args = parser.parse_args()
    
    # Chuyển đổi thành tuple
    target_size = tuple(args.target_size)
    bins = tuple(args.bins) if len(args.bins) > 1 else args.bins[0]
    data_dir = args.data_dir
    
    print("=" * 70)
    print("TRÍCH XUẤT ĐẶC TRƯNG HISTOGRAM - COLOR HISTOGRAM")
    print("=" * 70)
    print(f"\n📁 Dataset: {data_dir}")
    print(f"📐 Kích thước ảnh: {target_size}")
    print(f"🎨 Không gian màu: {args.color_space}")
    print(f"🔢 Bins: {bins}")
    print(f"📏 Normalize: {'BẬT' if args.normalize else 'TẮT'}")
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
    
    features_list, labels_list, class_names = extract_histogram_from_dataset(
        data_dir,
        target_size=target_size,
        color_space=args.color_space,
        bins=bins,
        normalize=args.normalize,
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
        
        # Giải thích kích thước
        if isinstance(bins, tuple):
            expected_size = bins[0] * bins[1] * bins[2] if len(bins) == 3 else bins[0]
        else:
            expected_size = bins
        print(f"   (= {bins} bins cho {args.color_space})")
        
        print("\n" + "=" * 70)
        print("✅ HOÀN THÀNH!")
        print("=" * 70)
    else:
        print("\n❌ Không trích xuất được đặc trưng nào!")

