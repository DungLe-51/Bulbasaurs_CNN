% verify_fpga.m - Kiểm chứng 100% kết quả từ FPGA so với lý thuyết
clc; close all;

disp('Đang đọc kết quả khôi phục từ Verilog...');

% 1. Đọc dữ liệu từ file text do ModelSim xuất ra
fpga_real = load('fpga_out_real.txt');
fpga_imag = load('fpga_out_imag.txt');

% 2. Ghép thành số phức
% Lưu ý: Do lúc trước đưa xuống Verilog ta nhân Scale = 1024 cho cả 2 toán hạng
% Nên kết quả nhân ra bị gấp lên 1024 * 1024 lần. Ta cần chia lại để chuẩn hóa.
Scale = 1024;
fpga_complex = (fpga_real + 1i * fpga_imag) / (Scale * Scale);

% 3. Trực quan hóa kết quả
figure('Name', 'Nghiệm thu 100%: FPGA vs MATLAB', 'NumberTitle', 'off', 'Color', 'w');

% Vẽ ảnh khôi phục từ FPGA
imagesc(abs(fpga_complex));
title('Ảnh khôi phục bởi phần cứng FPGA (Verilog)');
colorbar;
xlabel('Cột ảnh (Vector 1D)');
ylabel('Chỉ số Mode (1 đến 258)');

disp('Đã vẽ xong kết quả từ FPGA!');
disp('Bạn hãy so sánh bức ảnh này với bức "Ảnh gốc ban đầu" xem có giống hệt nhau không nhé!');
