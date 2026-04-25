import h5py

file_path = '20211102T152000_patch1_highnoise_30_100_0_22.h5'

try:
    with h5py.File(file_path, 'r') as f:
        print("=== BÁO CÁO CẤU TRÚC HDF5 ===")
        print("1. Kích thước (Shape) của 'patch':", f['patch'].shape)
        print("2. Kiểu dữ liệu (Dtype) của 'patch':", f['patch'].dtype)
        print("3. Kích thước (Shape) của 'mask':", f['mask'].shape)
        print("4. Kiểu dữ liệu (Dtype) của 'mask':", f['mask'].dtype)
        
        # In thử 5 giá trị đầu tiên để xem biên độ thực tế
        print("\n5. Giá trị mẫu của 'patch' (5 phần tử đầu):")
        print(f['patch'][0, :5] if len(f['patch'].shape) == 2 else f['patch'][0, 0, :5])
except Exception as e:
    print("Lỗi khi đọc file:", e)
