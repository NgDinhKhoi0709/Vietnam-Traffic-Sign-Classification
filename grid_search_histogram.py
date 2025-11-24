# -*- coding: utf-8 -*-
"""
Grid Search cho Histogram Features
Tìm tham số tốt nhất cho Histogram bằng cách đánh giá trên tập validation (20% từ train)
"""

import sys
import io
import numpy as np
import time
import pickle
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from itertools import product

# Cấu hình encoding UTF-8 cho Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from histogram_feature import extract_histogram_from_dataset


def grid_search_histogram():
    """
    Grid search để tìm tham số tốt nhất cho Histogram
    """
    print("=" * 80)
    print("GRID SEARCH - HISTOGRAM FEATURES")
    print("=" * 80)
    
    # Định nghĩa grid tham số cho Feature Extraction
    param_grid = {
        'target_size': [(64, 64), (128, 128), (256, 256)],
        'color_space': ['BGR', 'RGB', 'HSV', 'Lab'],
        'bins': [
            (8, 8, 8),
            (16, 16, 16),
            (18, 8, 8),    # Khuyến nghị cho HSV
            (32, 16, 16),
        ],
    }
    
    # Định nghĩa grid tham số cho SVM
    svm_param_grid = {
        'kernel': ['rbf', 'linear'],
        'C': [0.1, 1, 10],
        'gamma': ['scale']  # chỉ dùng với kernel='rbf'
    }
    
    print("\n📋 Grid tham số Feature Extraction:")
    for param, values in param_grid.items():
        print(f"   - {param}: {values}")
    
    print(f"\n📋 Grid tham số SVM:")
    for param, values in svm_param_grid.items():
        print(f"   - {param}: {values}")
    
    # Tính tổng số combinations
    feature_combinations = np.prod([len(v) for v in param_grid.values()])
    svm_combinations = len(svm_param_grid['kernel']) * len(svm_param_grid['C']) * len(svm_param_grid['gamma'])
    total_combinations = feature_combinations * svm_combinations
    
    print(f"\n📊 Feature combinations: {feature_combinations}")
    print(f"📊 SVM combinations: {svm_combinations}")
    print(f"📊 Tổng số combinations: {total_combinations}")
    
    # Tạo thư mục lưu kết quả
    results_dir = 'grid_search_results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Lưu kết quả
    results = []
    best_accuracy = 0
    best_params = None
    
    # Đếm combination
    current_combination = 0
    
    print("\n" + "=" * 80)
    print("BẮT ĐẦU GRID SEARCH")
    print("=" * 80)
    
    # Grid search - Nested loop cho Feature params và SVM params
    for target_size, color_space, bins in product(
        param_grid['target_size'],
        param_grid['color_space'],
        param_grid['bins']
    ):
        print(f"\n{'='*80}")
        print(f"Feature params: target_size={target_size}, color_space={color_space}, bins={bins}")
        print(f"{'='*80}")
        
        try:
            # 1. Trích xuất features từ train set (chỉ 1 lần cho mỗi feature params)
            print("   → Trích xuất features từ train set...")
            feature_start_time = time.time()
            
            X_train_full, y_train_full, class_names = extract_histogram_from_dataset(
                'vn-signs/train',
                target_size=target_size,
                color_space=color_space,
                bins=bins,
                normalize=True,
                use_cache=False  # Không dùng cache vì mỗi lần khác tham số
            )
            
            if len(X_train_full) == 0:
                print("   ❌ Không trích xuất được features, skip...")
                continue
            
            # Convert to numpy array
            X_train_full = np.array(X_train_full)
            
            # Encode labels
            label_encoder = LabelEncoder()
            y_train_full = label_encoder.fit_transform(y_train_full)
            
            # 2. Chia train/validation (80/20)
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_full, y_train_full,
                test_size=0.2,
                random_state=42,
                stratify=y_train_full
            )
            
            print(f"   → Train: {X_train.shape[0]} samples, Validation: {X_val.shape[0]} samples")
            print(f"   → Feature extraction time: {time.time() - feature_start_time:.2f}s")
            
            # 3. Chuẩn hóa (chỉ 1 lần)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # 4. Thử tất cả các SVM params
            for kernel, C, gamma in product(
                svm_param_grid['kernel'],
                svm_param_grid['C'],
                svm_param_grid['gamma']
            ):
                current_combination += 1
                
                # Gamma chỉ áp dụng cho RBF kernel
                if kernel == 'linear' and gamma not in ['scale', 'auto']:
                    continue
                
                print(f"\n   [{current_combination}/{total_combinations}] SVM: kernel={kernel}, C={C}, gamma={gamma}")
                
                try:
                    svm_start_time = time.time()
                    
                    # Train SVM
                    if kernel == 'linear':
                        svm = SVC(kernel=kernel, C=C, random_state=42)
                    else:  # rbf
                        svm = SVC(kernel=kernel, C=C, gamma=gamma, random_state=42)
                    
                    svm.fit(X_train_scaled, y_train)
                    
                    # Đánh giá trên validation
                    y_val_pred = svm.predict(X_val_scaled)
                    val_accuracy = accuracy_score(y_val, y_val_pred)
                    
                    # Đánh giá trên train
                    y_train_pred = svm.predict(X_train_scaled)
                    train_accuracy = accuracy_score(y_train, y_train_pred)
                    
                    svm_time = time.time() - svm_start_time
                    
                    print(f"      ✓ Train Accuracy: {train_accuracy*100:.2f}%")
                    print(f"      ✓ Val Accuracy:   {val_accuracy*100:.2f}%")
                    print(f"      ✓ SVM Time: {svm_time:.2f}s")
                    
                    # Lưu kết quả
                    result = {
                        'target_size': target_size,
                        'color_space': color_space,
                        'bins': bins,
                        'kernel': kernel,
                        'C': C,
                        'gamma': gamma if kernel == 'rbf' else None,
                        'train_accuracy': train_accuracy,
                        'val_accuracy': val_accuracy,
                        'time': svm_time,
                        'feature_dim': X_train.shape[1]
                    }
                    results.append(result)
                    
                    # Cập nhật best
                    if val_accuracy > best_accuracy:
                        best_accuracy = val_accuracy
                        best_params = result.copy()
                        print(f"      🏆 NEW BEST! Val Accuracy: {best_accuracy*100:.2f}%")
                
                except Exception as e:
                    print(f"      ❌ Lỗi SVM: {e}")
                    continue
            
        except Exception as e:
            print(f"   ❌ Lỗi feature extraction: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Hiển thị kết quả
    print("\n" + "=" * 80)
    print("KẾT QUẢ GRID SEARCH")
    print("=" * 80)
    
    if len(results) > 0:
        # Sắp xếp theo validation accuracy
        results_sorted = sorted(results, key=lambda x: x['val_accuracy'], reverse=True)
        
        print(f"\n🏆 TOP 10 BEST CONFIGURATIONS:")
        print("-" * 80)
        for i, result in enumerate(results_sorted[:10], 1):
            print(f"\n{i}. Val Accuracy: {result['val_accuracy']*100:.2f}% | "
                  f"Train Accuracy: {result['train_accuracy']*100:.2f}%")
            print(f"   Feature: target_size={result['target_size']}, color_space={result['color_space']}, bins={result['bins']}")
            print(f"   SVM: kernel={result['kernel']}, C={result['C']}, gamma={result['gamma']}")
            print(f"   Feature dim: {result['feature_dim']}, Time: {result['time']:.2f}s")
        
        # Lưu kết quả vào file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(results_dir, f'histogram_grid_search_{timestamp}.pkl')
        
        with open(result_file, 'wb') as f:
            pickle.dump({
                'results': results,
                'best_params': best_params,
                'param_grid': param_grid,
                'svm_param_grid': svm_param_grid
            }, f)
        
        print(f"\n💾 Đã lưu kết quả vào: {result_file}")
        
        # Lưu kết quả dạng text
        txt_file = os.path.join(results_dir, f'histogram_grid_search_{timestamp}.txt')
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write("GRID SEARCH RESULTS - HISTOGRAM FEATURES\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total combinations tested: {len(results)}\n\n")
            
            f.write("BEST CONFIGURATION:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Validation Accuracy: {best_params['val_accuracy']*100:.2f}%\n")
            f.write(f"Train Accuracy: {best_params['train_accuracy']*100:.2f}%\n")
            f.write(f"\nFeature Extraction Parameters:\n")
            f.write(f"  target_size: {best_params['target_size']}\n")
            f.write(f"  color_space: {best_params['color_space']}\n")
            f.write(f"  bins: {best_params['bins']}\n")
            f.write(f"\nSVM Parameters:\n")
            f.write(f"  kernel: {best_params['kernel']}\n")
            f.write(f"  C: {best_params['C']}\n")
            f.write(f"  gamma: {best_params['gamma']}\n")
            f.write(f"\nOther Info:\n")
            f.write(f"  Feature dimension: {best_params['feature_dim']}\n")
            f.write(f"  Time: {best_params['time']:.2f}s\n\n")
            
            f.write("\nALL RESULTS (sorted by validation accuracy):\n")
            f.write("-" * 80 + "\n")
            for i, result in enumerate(results_sorted, 1):
                f.write(f"\n{i}. Val Acc: {result['val_accuracy']*100:.2f}% | "
                       f"Train Acc: {result['train_accuracy']*100:.2f}%\n")
                f.write(f"   {result}\n")
        
        print(f"💾 Đã lưu báo cáo text: {txt_file}")
        
        print("\n" + "=" * 80)
        print("✅ HOÀN THÀNH GRID SEARCH!")
        print("=" * 80)
    else:
        print("\n❌ Không có kết quả nào!")


if __name__ == "__main__":
    grid_search_histogram()

