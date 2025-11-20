"""
Chiết xuất đặc trưng CCV (Color Coherence Vector)

Module này cung cấp các hàm để trích xuất đặc trưng CCV từ ảnh biển báo giao thông.
CCV là một phương pháp mô tả màu sắc của ảnh bằng cách phân loại các pixel màu
thành hai loại: coherent (liên kết) và incoherent (không liên kết).

Nguyên lý hoạt động:
1. Lượng tử hóa không gian màu thành số lượng bin nhất định
2. Xác định các vùng liên thông (connected components) cho mỗi màu
3. Phân loại pixel:
   - Coherent: thuộc vùng liên thông lớn (>= ngưỡng)
   - Incoherent: thuộc vùng liên thông nhỏ (< ngưỡng)
4. Tạo vector đặc trưng gồm số lượng pixel coherent và incoherent cho mỗi màu

Ưu điểm:
- Mô tả không chỉ phân bố màu mà còn cả không gian màu (spatial coherence)
- Phân biệt được các ảnh có histogram màu giống nhau nhưng cấu trúc khác nhau
- Hiệu quả cho bài toán phân loại ảnh dựa trên màu sắc
- Phù hợp với biển báo giao thông vì biển báo có màu sắc đặc trưng

Tham số quan trọng:
- n_bins: Số lượng bin cho mỗi kênh màu (thường là 8-16)
- threshold: Ngưỡng kích thước vùng để phân loại coherent/incoherent
"""

import cv2
import numpy as np
from scipy import ndimage
import os
import pickle
import argparse
from datetime import datetime
from tqdm import tqdm


def blur_and_quantize(image, n_bins=8, blur_size=7, color_space='BGR'):
    """
    Làm mờ và lượng tử hóa ảnh để chuẩn bị cho CCV
    
    Làm mờ ảnh giúp giảm nhiễu và tạo các vùng màu đồng nhất hơn,
    giúp việc xác định vùng liên thông chính xác hơn.
    
    Tham số:
    ----------
    image : numpy.ndarray
        Ảnh đầu vào (BGR format từ OpenCV)
    n_bins : int, mặc định=8
        Số lượng bin cho mỗi kênh màu
        Với n_bins=8, mỗi kênh sẽ có 8 mức, tổng số màu = 8^3 = 512
    blur_size : int, mặc định=7
        Kích thước kernel cho Gaussian blur (phải là số lẻ)
    color_space : str, mặc định='BGR'
        Không gian màu: 'BGR', 'RGB', 'HSV'
    
    Trả về:
    ----------
    quantized : numpy.ndarray
        Ảnh đã được lượng tử hóa, mỗi pixel là một chỉ số màu duy nhất
        Giá trị từ 0 đến (n_bins^3 - 1)
    """
    # Chuyển đổi không gian màu nếu cần
    if color_space == 'BGR':
        image_converted = image
    elif color_space == 'RGB':
        image_converted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif color_space == 'HSV':
        image_converted = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    else:
        raise ValueError(f"Không gian màu không hợp lệ: {color_space}")
    
    # Làm mờ ảnh để giảm nhiễu
    blurred = cv2.GaussianBlur(image_converted, (blur_size, blur_size), 0)
    
    # Lượng tử hóa mỗi kênh màu
    # Chia giá trị pixel (0-255) thành n_bins khoảng
    # Lưu ý: HSV có H từ 0-179, cần xử lý riêng
    quantized_channels = blurred.astype(np.int32)
    if color_space == 'HSV':
        # Kênh H: 0-179 -> chia theo n_bins
        quantized_channels[:, :, 0] = (blurred[:, :, 0] * n_bins // 180).astype(np.int32)
        # Kênh S, V: 0-255
        quantized_channels[:, :, 1] = (blurred[:, :, 1] // (256 // n_bins)).astype(np.int32)
        quantized_channels[:, :, 2] = (blurred[:, :, 2] // (256 // n_bins)).astype(np.int32)
    else:
        # BGR/RGB: tất cả kênh 0-255
        step = 256 // n_bins
        quantized_channels = (blurred // step).astype(np.int32)
    
    # Kết hợp 3 kênh thành một chỉ số duy nhất
    # Công thức: index = C1 * n_bins^2 + C2 * n_bins + C3
    quantized = (quantized_channels[:, :, 0] * n_bins * n_bins + 
                 quantized_channels[:, :, 1] * n_bins + 
                 quantized_channels[:, :, 2])
    
    return quantized


def extract_ccv_features(image, n_bins=8, threshold=100, blur_size=7, color_space='BGR'):
    """
    Trích xuất đặc trưng CCV từ một ảnh
    
    Tham số:
    ----------
    image : numpy.ndarray
        Ảnh đầu vào (BGR format từ OpenCV), shape=(height, width, 3)
    n_bins : int, mặc định=8
        Số lượng bin cho mỗi kênh màu
        Tổng số màu = n_bins^3
        Ví dụ: n_bins=8 → 512 màu, n_bins=16 → 4096 màu
    threshold : int, mặc định=100
        Ngưỡng kích thước vùng (số pixel) để phân loại coherent
        Vùng có >= threshold pixel được coi là coherent
        Vùng có < threshold pixel được coi là incoherent
    blur_size : int, mặc định=7
        Kích thước kernel Gaussian blur (phải là số lẻ)
        Blur giúp tạo vùng màu liên tục hơn
    color_space : str, mặc định='BGR'
        Không gian màu: 'BGR', 'RGB', 'HSV'
        HSV khuyến nghị cho biển báo vì bất biến với ánh sáng
    
    Trả về:
    ----------
    ccv : numpy.ndarray
        Vector đặc trưng CCV, shape=(n_bins^3 * 2,)
        Mỗi màu có 2 giá trị: [coherent_count, incoherent_count]
        Vector được chuẩn hóa về [0, 1] bằng cách chia cho tổng số pixel
    
    Ghi chú:
    ----------
    - Giá trị n_bins lớn hơn cho độ chi tiết màu cao hơn nhưng vector lớn hơn
    - Threshold nên được điều chỉnh dựa trên kích thước ảnh
      Ảnh lớn → threshold lớn, ảnh nhỏ → threshold nhỏ
    - Vector CCV đã được chuẩn hóa nên bất biến với kích thước ảnh
    - Với biển báo giao thông, nên dùng color_space='HSV'
    """
    # Kiểm tra ảnh đầu vào
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Ảnh đầu vào phải là ảnh màu 3 kênh (BGR hoặc RGB)")
    
    # Bước 1: Làm mờ và lượng tử hóa ảnh
    quantized = blur_and_quantize(image, n_bins, blur_size, color_space)
    
    # Bước 2: Khởi tạo CCV
    total_colors = n_bins ** 3
    ccv = np.zeros(total_colors * 2, dtype=np.float32)
    
    # Bước 3: Xử lý từng màu
    for color_idx in range(total_colors):
        # Tạo mask cho màu hiện tại
        color_mask = (quantized == color_idx).astype(np.uint8)
        
        # Nếu không có pixel nào có màu này, bỏ qua
        if color_mask.sum() == 0:
            continue
        
        # Tìm các vùng liên thông (connected components)
        # Sử dụng 8-connectivity (8 pixel lân cận)
        labeled, num_features = ndimage.label(color_mask)
        
        # Đếm số pixel trong mỗi vùng liên thông
        coherent_count = 0
        incoherent_count = 0
        
        for region_idx in range(1, num_features + 1):
            region_size = (labeled == region_idx).sum()
            
            # Phân loại dựa trên kích thước vùng
            if region_size >= threshold:
                coherent_count += region_size
            else:
                incoherent_count += region_size
        
        # Lưu vào CCV
        ccv[color_idx * 2] = coherent_count
        ccv[color_idx * 2 + 1] = incoherent_count
    
    # Bước 4: Chuẩn hóa CCV
    # Chia cho tổng số pixel để vector bất biến với kích thước ảnh
    total_pixels = image.shape[0] * image.shape[1]
    if total_pixels > 0:
        ccv = ccv / total_pixels
    
    return ccv


def extract_ccv_from_file(image_path, target_size=(128, 128), **kwargs):
    """
    Trích xuất đặc trưng CCV từ file ảnh
    
    Tham số:
    ----------
    image_path : str
        Đường dẫn đến file ảnh
    target_size : tuple, mặc định=(128, 128)
        Kích thước ảnh mục tiêu (chiều rộng, chiều cao) để resize
        Resize giúp đồng nhất kích thước và giảm thời gian xử lý
    **kwargs : dict
        Các tham số bổ sung cho hàm extract_ccv_features
        (n_bins, threshold, blur_size)
    
    Trả về:
    ----------
    ccv : numpy.ndarray hoặc None
        Vector đặc trưng CCV, hoặc None nếu không đọc được ảnh
    """
    # Đọc ảnh
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Không thể đọc ảnh: {image_path}")
        return None
    
    # Resize ảnh về kích thước cố định
    image_resized = cv2.resize(image, target_size)
    
    # Trích xuất đặc trưng CCV
    ccv = extract_ccv_features(image_resized, **kwargs)
    
    return ccv


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


def extract_ccv_from_dataset(data_dir, target_size=(128, 128), use_cache=False, cache_file=None, **kwargs):
    """
    Trích xuất đặc trưng CCV từ toàn bộ dataset
    
    Hàm này duyệt qua tất cả các thư mục con trong data_dir, mỗi thư mục con
    là một lớp (class) của biển báo. Trích xuất đặc trưng CCV cho tất cả ảnh
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
        Các tham số bổ sung cho hàm extract_ccv_features
        (n_bins, threshold, blur_size)
    
    Trả về:
    ----------
    features_list : list of numpy.ndarray
        Danh sách các vector đặc trưng CCV
    labels_list : list of str
        Danh sách các nhãn tương ứng với mỗi vector đặc trưng
    class_names : list of str
        Danh sách các tên lớp trong dataset
    
    Ví dụ:
    ----------
    >>> # Không dùng cache
    >>> features, labels, classes = extract_ccv_from_dataset('vn-signs/train')
    >>> 
    >>> # Sử dụng cache
    >>> features, labels, classes = extract_ccv_from_dataset(
    >>>     'vn-signs/train',
    >>>     use_cache=True,
    >>>     cache_file='features_cache/ccv_train.pkl'
    >>> )
    """
    # Tự động tạo tên file cache nếu không được cung cấp
    if use_cache and cache_file is None:
        # Tạo tên file từ data_dir và tham số
        dataset_name = os.path.basename(data_dir.rstrip('/\\'))
        params_str = f"{target_size[0]}x{target_size[1]}"
        params_str += f"_bins{kwargs.get('n_bins', 8)}"
        params_str += f"_th{kwargs.get('threshold', 100)}"
        cache_file = f"features_cache/ccv_{dataset_name}_{params_str}.pkl"
    
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
            
            ccv = extract_ccv_from_file(image_path, target_size, **kwargs)
            
            if ccv is not None:
                features_list.append(ccv)
                labels_list.append(class_name)
    
    print(f"\n=== Hoàn thành ===")
    print(f"Tổng số ảnh đã trích xuất: {len(features_list)}")
    
    # Lưu cache nếu được yêu cầu
    if use_cache and cache_file and len(features_list) > 0:
        save_features_to_cache(cache_file, features_list, labels_list, class_names, target_size, **kwargs)
    
    return features_list, labels_list, class_names


def compare_ccv_distance(ccv1, ccv2):
    """
    Tính khoảng cách giữa hai vector CCV
    
    Sử dụng khoảng cách L1 (Manhattan distance) để so sánh hai CCV.
    Khoảng cách nhỏ → hai ảnh có màu sắc và cấu trúc tương tự.
    
    Tham số:
    ----------
    ccv1, ccv2 : numpy.ndarray
        Hai vector CCV cần so sánh (phải có cùng kích thước)
    
    Trả về:
    ----------
    distance : float
        Khoảng cách L1 giữa hai vector
        Giá trị trong khoảng [0, 2], với 0 là giống hệt nhau
    """
    # Khoảng cách L1 (Manhattan distance)
    distance = np.sum(np.abs(ccv1 - ccv2))
    return distance


if __name__ == "__main__":
    # Thiết lập argument parser
    parser = argparse.ArgumentParser(
        description='Trích xuất đặc trưng CCV (Color Coherence Vector) từ dataset biển báo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--data-dir', type=str, default='vn-signs/train',
                        help='Đường dẫn đến thư mục dataset')
    parser.add_argument('--target-size', type=int, nargs=2, default=[128, 128],
                        metavar=('WIDTH', 'HEIGHT'),
                        help='Kích thước ảnh mục tiêu (width height)')
    parser.add_argument('--n-bins', type=int, default=8,
                        help='Số lượng bin cho mỗi kênh màu (8 → 512 màu)')
    parser.add_argument('--threshold', type=int, default=100,
                        help='Ngưỡng phân loại coherent/incoherent (số pixel)')
    parser.add_argument('--blur-size', type=int, default=7,
                        help='Kích thước kernel Gaussian blur (số lẻ)')
    parser.add_argument('--color-space', type=str, default='HSV',
                        choices=['BGR', 'RGB', 'HSV'],
                        help='Không gian màu')
    parser.add_argument('--use-cache', action='store_true', default=True,
                        help='Sử dụng cache để tăng tốc')
    parser.add_argument('--cache-file', type=str, default=None,
                        help='Đường dẫn file cache (tự động nếu không chỉ định)')
    
    args = parser.parse_args()
    
    # Chuyển đổi target_size thành tuple
    target_size = tuple(args.target_size)
    data_dir = args.data_dir
    
    print("=" * 70)
    print("TRÍCH XUẤT ĐẶC TRƯNG CCV - COLOR COHERENCE VECTOR")
    print("=" * 70)
    print(f"\n📁 Dataset: {data_dir}")
    print(f"📐 Kích thước ảnh: {target_size}")
    print(f"🎨 Không gian màu: {args.color_space}")
    print(f"🔢 Số bins: {args.n_bins} (→ {args.n_bins**3} màu)")
    print(f"📏 Threshold: {args.threshold} pixels")
    print(f"🌫️  Blur size: {args.blur_size}x{args.blur_size}")
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
    
    features_list, labels_list, class_names = extract_ccv_from_dataset(
        data_dir,
        target_size=target_size,
        n_bins=args.n_bins,
        threshold=args.threshold,
        blur_size=args.blur_size,
        color_space=args.color_space,
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
        print(f"✓ Tổng số chiều: {features_list[0].shape[0]} = {args.n_bins}³ × 2 (coherent + incoherent)")
        
        print("\n" + "=" * 70)
        print("✅ HOÀN THÀNH!")
        print("=" * 70)
    else:
        print("\n❌ Không trích xuất được đặc trưng nào!")
