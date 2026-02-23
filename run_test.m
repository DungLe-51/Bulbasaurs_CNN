% 1. Khai báo các thông số đầu vào (Inputs)
lambda = 1550e-9;      % Bước sóng hoạt động (ví dụ: 1550 nm)
D = 50e-6;             % Đường kính lõi sợi quang (ví dụ: 50 micro-mét)
NA = 0.22;             % Khẩu độ số (Numerical Aperture)
Length = [0.1];        % Chiều dài đoạn sợi quang (ví dụ: 0.1 mét)
Rho = [0.05];           % Bán kính cong (inf nghĩa là sợi thẳng, không bị uốn cong)
Theta = [0];           % Góc uốn cong (0 radian)
N = 32;                % Kích thước lưới ảnh đầu ra (N x N pixel)

% 2. Gọi hàm
disp('Bắt đầu mô phỏng...');
[ T, NMode, lmap, mmap, EHHEmap, propconst, Er, Ep, Ez, Hr, Hp, Hz, img_size ] = ...
    MMF_simTM_PIM( lambda, D, NA, Length, Rho, Theta, N );

% 3. Hiển thị kết quả
disp(['Mô phỏng thành công! Số lượng Modes tìm thấy: ', num2str(NMode)]);
figure; 
imagesc(abs(T)); 
title('Ma trận truyền dẫn (Transmission Matrix - T)');
colorbar;
