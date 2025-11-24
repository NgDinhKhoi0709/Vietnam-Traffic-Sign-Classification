# -*- coding: utf-8 -*-
"""
Phân tích kết quả Grid Search cho CCV Features
Huấn luyện và đánh giá từng bộ tham số SVM tốt nhất trên tập test
"""

import sys
import io
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Cấu hình encoding UTF-8 cho Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from ccv_feature import extract_ccv_from_dataset


def analyze_grid_search_results(result_file):
    """
    Đọc và phân tích kết quả grid search
    
    Args:
        result_file: Đường dẫn đến file .pkl chứa kết quả
    
    Returns:
        svm_best_results_sorted: Danh sách các bộ tham số SVM tốt nhất
    """
    print("=" * 100)
    print("PHÂN TÍCH KẾT QUẢ GRID SEARCH - CCV FEATURES")
    print("=" * 100)
    
    # Đọc file kết quả
    try:
        with open(result_file, 'rb') as f:
            data = pickle.load(f)
        
        results = data['results']
        best_params = data['best_params']
        param_grid = data.get('param_grid', {})
        svm_param_grid = data.get('svm_param_grid', {})
        
        print(f"\n📁 File: {result_file}")
        print(f"📊 Tổng số kết quả: {len(results)}")
        
    except Exception as e:
        print(f"❌ Lỗi đọc file: {e}")
        return None
    
    # ========================================================================
    # PHÂN TÍCH TỔNG QUAN
    # ========================================================================
    print("\n" + "=" * 100)
    print("PHÂN TÍCH TỔNG QUAN")
    print("=" * 100)
    
    # Nhóm kết quả theo tham số SVM
    svm_to_ccv = defaultdict(list)
    
    for result in results:
        kernel = result['kernel']
        C = result['C']
        gamma = result.get('gamma', None)  # CCV có thể có gamma=None cho linear kernel
        svm_key = (kernel, C, gamma)
        
        ccv_params = {
            'target_size': result['target_size'],
            'n_bins': result['n_bins'],
            'threshold': result['threshold'],
            'color_space': result['color_space'],
            'val_accuracy': result['val_accuracy'],
            'train_accuracy': result['train_accuracy'],
            'feature_dim': result['feature_dim'],
            'time': result['time']
        }
        
        svm_to_ccv[svm_key].append(ccv_params)
    
    # Tập hợp các tham số SVM duy nhất
    svm_params_set = set()
    for result in results:
        kernel = result['kernel']
        C = result['C']
        gamma = result.get('gamma', None)
        svm_params_set.add((kernel, C, gamma))
    
    svm_params_list = sorted(list(svm_params_set))
    
    # Tìm best result cho mỗi tập tham số SVM
    print("\n🏆 KẾT QUẢ TỐT NHẤT CHO MỖI TẬP THAM SỐ SVM:")
    print("-" * 100)
    
    svm_best_results = []
    for svm_key in svm_params_list:
        ccv_list = svm_to_ccv[svm_key]
        best_ccv = max(ccv_list, key=lambda x: x['val_accuracy'])
        
        svm_best_results.append({
            'kernel': svm_key[0],
            'C': svm_key[1],
            'gamma': svm_key[2],
            'best_val_acc': best_ccv['val_accuracy'],
            'best_ccv': best_ccv
        })
    
    # Sắp xếp theo val_accuracy
    svm_best_results_sorted = sorted(svm_best_results, key=lambda x: x['best_val_acc'], reverse=True)
    
    for i, result in enumerate(svm_best_results_sorted, 1):
        gamma_str = str(result['gamma']) if result['gamma'] is not None else 'None'
        print(f"\n{i:2d}. SVM: kernel={result['kernel']:8s} | C={result['C']:6.1f} | gamma={gamma_str}")
        print(f"    Best Val Acc: {result['best_val_acc']*100:.2f}%")
        print(f"    Best CCV: target_size={result['best_ccv']['target_size']}, "
              f"n_bins={result['best_ccv']['n_bins']}, "
              f"threshold={result['best_ccv']['threshold']}, "
              f"color_space={result['best_ccv']['color_space']}")
    
    return svm_best_results_sorted


def train_and_evaluate_svm_config(config, config_idx, total_configs, results_base_dir='ccv_results'):
    """
    Huấn luyện và đánh giá một cấu hình SVM trên tập test
    
    Args:
        config: Dictionary chứa thông tin cấu hình SVM và CCV
        config_idx: Chỉ số cấu hình (1-based)
        total_configs: Tổng số cấu hình
        results_base_dir: Thư mục gốc lưu kết quả
    """
    print("\n" + "=" * 100)
    print(f"[{config_idx}/{total_configs}] HUẤN LUYỆN VÀ ĐÁNH GIÁ")
    print("=" * 100)
    
    # Lấy tham số
    kernel = config['kernel']
    C = config['C']
    gamma = config['gamma']
    ccv_params = config['best_ccv']
    
    target_size = ccv_params['target_size']
    n_bins = ccv_params['n_bins']
    threshold = ccv_params['threshold']
    color_space = ccv_params['color_space']
    
    print(f"\n📋 SVM Parameters:")
    print(f"   kernel: {kernel}")
    print(f"   C:      {C}")
    print(f"   gamma:  {gamma}")
    
    print(f"\n📋 CCV Parameters:")
    print(f"   target_size:  {target_size}")
    print(f"   n_bins:       {n_bins}")
    print(f"   threshold:    {threshold}")
    print(f"   color_space:  {color_space}")
    
    # Tạo tên thư mục
    if kernel == 'linear':
        folder_name = f"svm_kernel-{kernel}_C-{C}"
    else:
        gamma_str = str(gamma) if gamma is not None else 'None'
        folder_name = f"svm_kernel-{kernel}_C-{C}_gamma-{gamma_str}"
    
    result_dir = os.path.join(results_base_dir, folder_name)
    os.makedirs(result_dir, exist_ok=True)
    
    print(f"\n📁 Thư mục kết quả: {result_dir}")
    
    try:
        # ====================================================================
        # 1. TRÍCH XUẤT FEATURES
        # ====================================================================
        print("\n" + "-" * 100)
        print("BƯỚC 1: TRÍCH XUẤT FEATURES")
        print("-" * 100)
        
        # Train set
        print("\n→ Trích xuất features từ train set...")
        X_train, y_train, class_names = extract_ccv_from_dataset(
            'vn-signs/train',
            target_size=target_size,
            n_bins=n_bins,
            threshold=threshold,
            color_space=color_space,
            use_cache=False
        )
        
        X_train = np.array(X_train)
        print(f"   ✓ Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        
        # Test set
        print("\n→ Trích xuất features từ test set...")
        X_test, y_test, _ = extract_ccv_from_dataset(
            'vn-signs/test',
            target_size=target_size,
            n_bins=n_bins,
            threshold=threshold,
            color_space=color_space,
            use_cache=False
        )
        
        X_test = np.array(X_test)
        print(f"   ✓ Test:  {X_test.shape[0]} samples, {X_test.shape[1]} features")
        
        # ====================================================================
        # 2. ENCODE LABELS
        # ====================================================================
        print("\n" + "-" * 100)
        print("BƯỚC 2: ENCODE LABELS")
        print("-" * 100)
        
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)
        
        print(f"   ✓ Classes: {list(label_encoder.classes_)}")
        print(f"   ✓ Number of classes: {len(label_encoder.classes_)}")
        
        # Lưu label encoder
        le_path = os.path.join(result_dir, 'label_encoder.pkl')
        joblib.dump(label_encoder, le_path)
        print(f"   ✓ Đã lưu label encoder: {le_path}")
        
        # ====================================================================
        # 3. CHUẨN HÓA
        # ====================================================================
        print("\n" + "-" * 100)
        print("BƯỚC 3: CHUẨN HÓA")
        print("-" * 100)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"   ✓ Đã chuẩn hóa train và test sets")
        
        # Lưu scaler
        scaler_path = os.path.join(result_dir, 'scaler.pkl')
        joblib.dump(scaler, scaler_path)
        print(f"   ✓ Đã lưu scaler: {scaler_path}")
        
        # ====================================================================
        # 4. HUẤN LUYỆN SVM
        # ====================================================================
        print("\n" + "-" * 100)
        print("BƯỚC 4: HUẤN LUYỆN SVM")
        print("-" * 100)
        
        import time
        start_time = time.time()
        
        if kernel == 'linear':
            svm = SVC(kernel=kernel, C=C, random_state=42)
        else:
            if gamma is not None:
                svm = SVC(kernel=kernel, C=C, gamma=gamma, random_state=42)
            else:
                svm = SVC(kernel=kernel, C=C, random_state=42)
        
        svm.fit(X_train_scaled, y_train_encoded)
        
        training_time = time.time() - start_time
        print(f"   ✓ Thời gian huấn luyện: {training_time:.2f}s")
        
        # Lưu model
        model_path = os.path.join(result_dir, 'svm_model.pkl')
        joblib.dump(svm, model_path)
        print(f"   ✓ Đã lưu model: {model_path}")
        
        # ====================================================================
        # 5. DỰ ĐOÁN VÀ ĐÁNH GIÁ
        # ====================================================================
        print("\n" + "-" * 100)
        print("BƯỚC 5: DỰ ĐOÁN VÀ ĐÁNH GIÁ")
        print("-" * 100)
        
        # Dự đoán trên train
        y_train_pred = svm.predict(X_train_scaled)
        train_accuracy = accuracy_score(y_train_encoded, y_train_pred)
        
        # Dự đoán trên test
        y_test_pred = svm.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test_encoded, y_test_pred)
        
        print(f"\n   📊 Train Accuracy: {train_accuracy*100:.2f}%")
        print(f"   📊 Test Accuracy:  {test_accuracy*100:.2f}%")
        
        # ====================================================================
        # 6. LƯU CLASSIFICATION REPORT
        # ====================================================================
        print("\n" + "-" * 100)
        print("BƯỚC 6: LƯU CLASSIFICATION REPORT")
        print("-" * 100)
        
        report = classification_report(
            y_test_encoded, 
            y_test_pred, 
            target_names=label_encoder.classes_,
            digits=4
        )
        
        print("\n" + report)
        
        # Lưu vào file
        report_path = os.path.join(result_dir, 'classification_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("CLASSIFICATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"SVM Parameters:\n")
            f.write(f"  kernel: {kernel}\n")
            f.write(f"  C:      {C}\n")
            f.write(f"  gamma:  {gamma}\n\n")
            f.write(f"CCV Parameters:\n")
            f.write(f"  target_size:  {target_size}\n")
            f.write(f"  n_bins:       {n_bins}\n")
            f.write(f"  threshold:    {threshold}\n")
            f.write(f"  color_space:  {color_space}\n\n")
            f.write(f"Results:\n")
            f.write(f"  Train Accuracy: {train_accuracy*100:.2f}%\n")
            f.write(f"  Test Accuracy:  {test_accuracy*100:.2f}%\n")
            f.write(f"  Training Time:  {training_time:.2f}s\n\n")
            f.write(report)
        
        print(f"   ✓ Đã lưu classification report: {report_path}")
        
        # ====================================================================
        # 7. VẼ VÀ LƯU CONFUSION MATRIX
        # ====================================================================
        print("\n" + "-" * 100)
        print("BƯỚC 7: VẼ VÀ LƯU CONFUSION MATRIX")
        print("-" * 100)
        
        cm = confusion_matrix(y_test_encoded, y_test_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=label_encoder.classes_,
                    yticklabels=label_encoder.classes_)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title(f'Confusion Matrix\nkernel={kernel}, C={C}, gamma={gamma}')
        plt.tight_layout()
        
        cm_path = os.path.join(result_dir, 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Đã lưu confusion matrix: {cm_path}")
        
        # Lưu confusion matrix data
        cm_data_path = os.path.join(result_dir, 'confusion_matrix.pkl')
        with open(cm_data_path, 'wb') as f:
            pickle.dump({
                'confusion_matrix': cm,
                'class_names': label_encoder.classes_
            }, f)
        
        print(f"   ✓ Đã lưu confusion matrix data: {cm_data_path}")
        
        # ====================================================================
        # 8. LƯU ẢNH PHÂN LOẠI SAI
        # ====================================================================
        print("\n" + "-" * 100)
        print("BƯỚC 8: LƯU ẢNH PHÂN LOẠI SAI")
        print("-" * 100)
        
        # Tìm các ảnh dự đoán sai
        wrong_indices = np.where(y_test_pred != y_test_encoded)[0]
        
        print(f"\n   📊 Tổng số ảnh test: {len(y_test_encoded)}")
        print(f"   📊 Số ảnh dự đoán SAI: {len(wrong_indices)}")
        print(f"   📊 Số ảnh dự đoán ĐÚNG: {len(y_test_encoded) - len(wrong_indices)}")
        print(f"   📊 Tỷ lệ sai: {len(wrong_indices)/len(y_test_encoded)*100:.2f}%")
        
        if len(wrong_indices) > 0:
            # Tạo thư mục misclassified_images
            misclassified_dir = os.path.join(result_dir, 'misclassified_images')
            os.makedirs(misclassified_dir, exist_ok=True)
            
            # Đọc ảnh gốc từ test set
            import cv2
            test_dir = 'vn-signs/test'
            
            # Tạo mapping từ index đến file path
            test_image_paths = []
            test_labels = []
            
            for class_name in sorted(os.listdir(test_dir)):
                class_path = os.path.join(test_dir, class_name)
                if not os.path.isdir(class_path):
                    continue
                
                for img_name in sorted(os.listdir(class_path)):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        test_image_paths.append(os.path.join(class_path, img_name))
                        test_labels.append(class_name)
            
            # Lưu từng ảnh sai
            for idx in wrong_indices:
                true_label = label_encoder.classes_[y_test_encoded[idx]]
                pred_label = label_encoder.classes_[y_test_pred[idx]]
                
                # Đọc ảnh gốc
                img_path = test_image_paths[idx]
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Vẽ ảnh với nhãn
                fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                ax.imshow(img)
                ax.axis('off')
                ax.set_title(f'True: {true_label}\nPredicted: {pred_label}', 
                           fontsize=12, color='red', weight='bold')
                
                # Tên file
                img_filename = f"img{idx:03d}_true-{true_label}_pred-{pred_label}.png"
                save_path = os.path.join(misclassified_dir, img_filename)
                
                plt.tight_layout()
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
            
            print(f"   ✓ Đã lưu {len(wrong_indices)} ảnh vào: {misclassified_dir}")
        else:
            print(f"   🎉 Không có ảnh nào bị dự đoán sai!")
        
        # ====================================================================
        # 9. LƯU THÔNG TIN TỔNG HỢP
        # ====================================================================
        print("\n" + "-" * 100)
        print("BƯỚC 9: LƯU THÔNG TIN TỔNG HỢP")
        print("-" * 100)
        
        summary_path = os.path.join(result_dir, 'summary.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("TỔNG HỢP KẾT QUẢ\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("SVM Parameters:\n")
            f.write(f"  kernel: {kernel}\n")
            f.write(f"  C:      {C}\n")
            f.write(f"  gamma:  {gamma}\n\n")
            
            f.write("CCV Parameters:\n")
            f.write(f"  target_size:  {target_size}\n")
            f.write(f"  n_bins:       {n_bins}\n")
            f.write(f"  threshold:    {threshold}\n")
            f.write(f"  color_space:  {color_space}\n\n")
            
            f.write("Dataset:\n")
            f.write(f"  Train samples: {len(X_train)}\n")
            f.write(f"  Test samples:  {len(X_test)}\n")
            f.write(f"  Features:      {X_train.shape[1]}\n")
            f.write(f"  Classes:       {len(label_encoder.classes_)}\n\n")
            
            f.write("Results:\n")
            f.write(f"  Train Accuracy: {train_accuracy*100:.2f}%\n")
            f.write(f"  Test Accuracy:  {test_accuracy*100:.2f}%\n")
            f.write(f"  Training Time:  {training_time:.2f}s\n\n")
            
            f.write("Misclassified:\n")
            f.write(f"  Total test:     {len(y_test_encoded)}\n")
            f.write(f"  Correct:        {len(y_test_encoded) - len(wrong_indices)}\n")
            f.write(f"  Wrong:          {len(wrong_indices)}\n")
            f.write(f"  Error rate:     {len(wrong_indices)/len(y_test_encoded)*100:.2f}%\n")
        
        print(f"   ✓ Đã lưu summary: {summary_path}")
        
        print("\n" + "=" * 100)
        print(f"✅ HOÀN THÀNH CẤU HÌNH [{config_idx}/{total_configs}]")
        print("=" * 100)
        
        return {
            'folder': folder_name,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'training_time': training_time,
            'num_misclassified': len(wrong_indices)
        }
        
    except Exception as e:
        print(f"\n❌ LỖI: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """
    Hàm main
    """
    # Đường dẫn đến file kết quả
    result_file = 'grid_search_results/ccv_grid_search_20251124_015948.pkl'
    
    # Kiểm tra file tồn tại
    if not os.path.exists(result_file):
        print(f"❌ File không tồn tại: {result_file}")
        print("\n💡 Vui lòng cung cấp đường dẫn đúng đến file .pkl")
        
        # Tìm các file .pkl trong thư mục grid_search_results
        results_dir = 'grid_search_results'
        if os.path.exists(results_dir):
            pkl_files = [f for f in os.listdir(results_dir) if f.endswith('.pkl')]
            if pkl_files:
                print(f"\n📁 Các file .pkl có sẵn trong {results_dir}:")
                for i, f in enumerate(pkl_files, 1):
                    print(f"   {i}. {f}")
        return
    
    # ========================================================================
    # BƯỚC 1: PHÂN TÍCH KẾT QUẢ GRID SEARCH
    # ========================================================================
    svm_best_results = analyze_grid_search_results(result_file)
    
    if svm_best_results is None:
        print("\n❌ Không thể phân tích kết quả!")
        return
    
    # ========================================================================
    # BƯỚC 2: TẠO THỦ MỤC CCV_RESULTS
    # ========================================================================
    print("\n" + "=" * 100)
    print("TẠO THỦ MỤC CCV_RESULTS")
    print("=" * 100)
    
    results_base_dir = 'ccv_results'
    os.makedirs(results_base_dir, exist_ok=True)
    print(f"\n✓ Đã tạo thư mục: {results_base_dir}")
    
    # ========================================================================
    # BƯỚC 3: HUẤN LUYỆN VÀ ĐÁNH GIÁ TỪNG CẤU HÌNH
    # ========================================================================
    print("\n" + "=" * 100)
    print("HUẤN LUYỆN VÀ ĐÁNH GIÁ TỪNG CẤU HÌNH SVM")
    print("=" * 100)
    
    total_configs = len(svm_best_results)
    print(f"\n📊 Tổng số cấu hình cần huấn luyện: {total_configs}")
    
    all_results = []
    
    for i, config in enumerate(svm_best_results, 1):
        result = train_and_evaluate_svm_config(
            config, 
            config_idx=i, 
            total_configs=total_configs,
            results_base_dir=results_base_dir
        )
        
        if result is not None:
            all_results.append(result)
    
    # ========================================================================
    # BƯỚC 4: TỔNG HỢP KẾT QUẢ
    # ========================================================================
    print("\n" + "=" * 100)
    print("TỔNG HỢP KẾT QUẢ TẤT CẢ CẤU HÌNH")
    print("=" * 100)
    
    if len(all_results) > 0:
        # Sắp xếp theo test accuracy
        all_results_sorted = sorted(all_results, key=lambda x: x['test_accuracy'], reverse=True)
        
        print(f"\n🏆 BẢNG XẾP HẠNG THEO TEST ACCURACY:")
        print("-" * 100)
        print(f"{'#':<4} {'Folder':<50} {'Train Acc':<12} {'Test Acc':<12} {'Time':<10} {'Errors':<8}")
        print("-" * 100)
        
        for i, result in enumerate(all_results_sorted, 1):
            print(f"{i:<4} {result['folder']:<50} "
                  f"{result['train_accuracy']*100:>10.2f}% "
                  f"{result['test_accuracy']*100:>10.2f}% "
                  f"{result['training_time']:>8.2f}s "
                  f"{result['num_misclassified']:>6}")
        
        # Lưu tổng hợp
        summary_all_path = os.path.join(results_base_dir, 'summary_all_configs.txt')
        with open(summary_all_path, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write("TỔNG HỢP KẾT QUẢ TẤT CẢ CẤU HÌNH SVM\n")
            f.write("=" * 100 + "\n\n")
            f.write(f"Ngày: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Tổng số cấu hình: {len(all_results)}\n\n")
            
            f.write("BẢNG XẾP HẠNG THEO TEST ACCURACY:\n")
            f.write("-" * 100 + "\n")
            f.write(f"{'#':<4} {'Folder':<50} {'Train Acc':<12} {'Test Acc':<12} {'Time':<10} {'Errors':<8}\n")
            f.write("-" * 100 + "\n")
            
            for i, result in enumerate(all_results_sorted, 1):
                f.write(f"{i:<4} {result['folder']:<50} "
                       f"{result['train_accuracy']*100:>10.2f}% "
                       f"{result['test_accuracy']*100:>10.2f}% "
                       f"{result['training_time']:>8.2f}s "
                       f"{result['num_misclassified']:>6}\n")
            
            f.write("\n" + "=" * 100 + "\n")
            f.write(f"🏆 BEST CONFIGURATION:\n")
            f.write("-" * 100 + "\n")
            best = all_results_sorted[0]
            f.write(f"Folder:         {best['folder']}\n")
            f.write(f"Train Accuracy: {best['train_accuracy']*100:.2f}%\n")
            f.write(f"Test Accuracy:  {best['test_accuracy']*100:.2f}%\n")
            f.write(f"Training Time:  {best['training_time']:.2f}s\n")
            f.write(f"Misclassified:  {best['num_misclassified']}\n")
        
        print(f"\n💾 Đã lưu tổng hợp: {summary_all_path}")
        
        print("\n" + "=" * 100)
        print("✅ HOÀN THÀNH TẤT CẢ!")
        print("=" * 100)
        print(f"\n📁 Tất cả kết quả đã được lưu trong: {results_base_dir}/")
        print(f"   - Mỗi cấu hình SVM có 1 thư mục riêng")
        print(f"   - Mỗi thư mục chứa:")
        print(f"     + Model (svm_model.pkl)")
        print(f"     + Label encoder (label_encoder.pkl)")
        print(f"     + Scaler (scaler.pkl)")
        print(f"     + Classification report (classification_report.txt)")
        print(f"     + Confusion matrix (confusion_matrix.png, .pkl)")
        print(f"     + Misclassified images (misclassified_images/)")
        print(f"     + Summary (summary.txt)")
    else:
        print("\n❌ Không có kết quả nào!")


if __name__ == "__main__":
    main()

