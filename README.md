# DeepDAS-FPGA: Hardware Acceleration for 1D CNN Seismic Detection

## Project Overview
Dự án **DeepDAS** nhằm mục đích triển khai mô hình học sâu (1D CNN) lên phần cứng FPGA để phát hiện tín hiệu động đất và sóng thần từ hệ thống Cáp quang Cảm biến Phân tán (Distributed Acoustic Sensing - DAS). 

Mục tiêu cốt lõi: Đưa mô hình AI từ môi trường Python/PyTorch xuống chạy thực tế trên board Xilinx Virtex-7 (VC707) thông qua giao tiếp PCIe tốc độ cao, đạt độ trễ siêu thấp.

## Kiến trúc Hệ thống (System Architecture)
Hệ thống được chia làm 3 tầng rõ rệt:
1. **Host PC (Ubuntu):** Quản lý luồng dữ liệu sóng địa chấn và giao tiếp qua khe cắm PCIe.
2. **XDMA (PCIe Bridge):** Cầu nối phần cứng trên FPGA, nhận dữ liệu từ Host PC qua chuẩn AXI-Stream (H2C) và trả kết quả về (C2H).
3. **FINN AI Core:** Bộ não nội suy (Inference Engine) thuần Dataflow, được đúc tự động từ mạng nơ-ron bằng Xilinx FINN. Không cần CPU điều khiển, chạy tự động khi có luồng dữ liệu (Stream).


## 📂 Cấu trúc Kho lưu trữ (Repository Structure)
Kho lưu trữ này tuân thủ nguyên tắc "Chỉ lưu Code và Cấu trúc". Rác biên dịch và file data dung lượng lớn đã bị chặn hoàn toàn.

📦 Bulbasaurs_CNN
 ┣ 📂 V2/                     # [GIAI ĐOẠN 1] Code AI & Kịch bản đúc Chip
 ┃ ┣ 📜 models_brevitas.py    # Định nghĩa cấu trúc mạng 1D CNN (Lượng tử hóa QAT)
 ┃ ┣ 📜 train_qat.py          # Script huấn luyện AI
 ┃ ┣ 📜 export_2d.py          # Script chuyển đổi sang chuẩn QONNX
 ┃ ┗ 📜 build.py              # Script điều khiển FINN Compiler đúc ra IP Core
 ┣ 📂 DeepDAS_PCIe_SoC/       # [GIAI ĐOẠN 3] Dự án Phần cứng Vivado 2025.2
 ┃ ┣ 📜 DeepDAS_PCIe_SoC.xpr  # File cấu hình Project Vivado chính
 ┃ ┗ 📂 DeepDAS_PCIe_SoC.srcs/# Chứa bản vẽ Block Design (deepdas_top.bd) và Wrapper
 ┗ 📜 .gitignore              # Lưới lọc thép chặn file rác và file > 100MB
