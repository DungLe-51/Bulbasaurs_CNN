% --- File: gen_verilog_data.m ---
% Mục đích: Tạo thuốc giải (Tikinv) và xuất dữ liệu test cho Verilog
clc; close all;

%% 1. Kế thừa dữ liệu từ Workspace của bạn
% Do bạn vừa chạy xong run_test, ma trận T và NMode đã có sẵn trong bộ nhớ.
if ~exist('T', 'var')
    error('Hãy chạy file run_test.m với Rho=0.05 trước khi chạy file này!');
end

disp('1. Bắt đầu tính Ma trận khôi phục (Thuốc giải)...');

%% 2. Dùng Tikinv.m để tính Ma trận Nghịch đảo
p = 0.05; % Hệ số chống nhiễu (Regularization)
[T_inv, ~] = Tikinv(T, p); 

%% 3. Tạo dữ liệu Test (Mô phỏng chụp 1 bức ảnh)
disp('2. Đang tạo dữ liệu ảnh đầu vào và ảnh nhiễu...');
% Tạo một "bức ảnh" gốc ngẫu nhiên (dưới dạng hệ số mode)
Input_Image = rand(NMode, 1) + 1i*rand(NMode, 1);
Input_Image = Input_Image / max(abs(Input_Image)); % Chuẩn hóa

% Truyền qua sợi quang cong để tạo Speckle (Nhiễu hạt)
Speckle_Out = T * Input_Image; 

% Dùng "Thuốc giải" để khôi phục lại thử trên MATLAB xem thành công không
Recovered_Image = T_inv * Speckle_Out;

% Vẽ hình kiểm chứng
figure('Name', 'Kiểm chứng thuật toán Khôi phục');
subplot(1,3,1); imagesc(abs(Input_Image)); title('1. Ảnh gốc ban đầu');
subplot(1,3,2); imagesc(abs(Speckle_Out)); title('2. Ảnh nhiễu (Speckle)');
subplot(1,3,3); imagesc(abs(Recovered_Image)); title('3. Ảnh khôi phục (T^{-1})');

%% 4. Ép kiểu và Xuất file cho VERILOG
disp('3. Đang xuất file .mem cho FPGA Verilog...');
Scale = 1024; % Nhân hệ số 2^10 để chuyển số thực thành số nguyên cho Verilog

% Hàm phụ trợ chuyển đổi
to_fixed = @(x) typecast(int16(round(x * Scale)), 'uint16');

% Mở file
fid_Tr = fopen('T_inv_real.mem', 'w'); fid_Ti = fopen('T_inv_imag.mem', 'w');
fid_Sr = fopen('Speckle_real.mem', 'w'); fid_Si = fopen('Speckle_imag.mem', 'w');

% Ghi ma trận T_inv (Quét từng hàng)
for r = 1:NMode
    for c = 1:NMode
        fprintf(fid_Tr, '%04X\n', to_fixed(real(T_inv(r,c))));
        fprintf(fid_Ti, '%04X\n', to_fixed(imag(T_inv(r,c))));
    end
end

% Ghi Vector Speckle
for i = 1:NMode
    fprintf(fid_Sr, '%04X\n', to_fixed(real(Speckle_Out(i))));
    fprintf(fid_Si, '%04X\n', to_fixed(imag(Speckle_Out(i))));
end

fclose('all');
disp('HOÀN THÀNH 100%! Đã có 4 file .mem sẵn sàng cho Verilog.');
