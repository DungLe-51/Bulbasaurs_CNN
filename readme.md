# CNN_AGAiN

`CNN_AGAiN` là một thiết kế RTL SystemVerilog cho bộ xử lý CNN 1D mục đích tổng hợp trên FPGA bằng Vivado.

## Mục tiêu
- Thiết kế một bộ lõi CNN 1D có thể thực hiện:
  - `CONV1D` (1D convolution)
  - `DWCONV1D` (depthwise convolution 1D)
  - `MAXPOOL1D`
- Hỗ trợ cấu hình bằng thanh ghi mô tả lớp và giao tiếp bộ nhớ chủ (host interface).
- Sử dụng kiến trúc bộ nhớ ping-pong cho IFM/OFM và mô-đun SRAM cho trọng số và tham số.

## Cấu trúc dự án
- `CNN_AGAiN.xpr`: Dự án Vivado chính.
- `CNN_AGAiN.runs/`: Kết quả tổng hợp, đặt chỗ, định tuyến và báo cáo.
- `CNN_AGAiN.srcs/sources_1/imports/new/`: Thư mục chứa mã RTL SystemVerilog.
- `CNN_AGAiN.hw/` và `CNN_AGAiN.sim/`: Thông tin phần cứng và mô phỏng liên quan.

## Thành phần chính
### 1. Lõi chính
- `cnn1D_core_top.sv`
  - Mô-đun top-level của bộ xử lý CNN 1D.
  - Nhận cấu hình qua `desc_*` và giao tiếp dữ liệu qua `host_*`.
  - Kết hợp các khối con: `layer_desc_regs`, `cnn1d_controller`, `bank_loader`, các bank nhớ, `pe_array`, `requantize`, `activation_relu`, `maxpool1d`, và `writeback_unit`.

### 2. Bộ điều khiển
- `cnn1d_controller.sv`
  - FSM điều khiển toàn bộ quá trình tính toán.
  - Quản lý trạng thái: xác thực cấu hình, đọc bias, khởi tạo bộ cộng, đọc MAC, tích lũy, hậu xử lý và ghi OFM.
  - Hỗ trợ các phép toán: `CONV1D`, `DWCONV1D`, `MAXPOOL1D`.

### 3. Cấu hình lớp
- `layer_desc.sv`
  - Lưu thanh ghi mô tả thuộc tính lớp.
  - Buffer đường ống các thông số mô tả để tránh thay đổi trong khi tính toán.

### 4. Bộ nhớ và arbiter
- `bank_loader.sv`
  - Phân loại truy cập host đến IFM, WGT, PARAM và OFM.
- `bank_arbiter.sv`
  - Chọn dữ liệu trả về phù hợp cho host dựa trên `host_sel_i`.
- `ifm_pingpong_bank.sv` / `ofm_pingpong_bank.sv`
  - Thiết kế ping-pong để host truy cập và lõi xử lý truy cập độc lập.
- `weight_sram_bank.sv`
  - SRAM lưu trọng số.
- `bias_scale_bank.sv`
  - SRAM lưu bias, hệ số nhân và dịch dịch bit cho quá trình tái lượng tử.

### 5. Sinh địa chỉ
- `ifm_addr_gen.sv`
  - Sinh địa chỉ đọc IFM theo vị trí và kênh.
- `ofm_addr_gen.sv`
  - Sinh địa chỉ ghi OFM theo vị trí và kênh.
- `weight_addr_gen.sv`
  - Sinh địa chỉ đọc trọng số cho cả conv và depthwise conv.

### 6. Tính toán MAC và hậu xử lý
- `pe_array.sv` / `pe_lane.sv`
  - Mặc dù hiện tại dòng chính dùng 1 lane, mô-đun này là cơ sở cho khối xử lý phần tử (PE).
- `mac_int8.sv`
  - Thực hiện nhân tích với dữ liệu 8-bit.
- `requantize.sv` / `quanitize.sv`
  - Đổi kết quả accumulator sang kích thước dữ liệu đầu ra.
- `activation_relu.sv`
  - Áp dụng ReLU tùy chọn sau khi tái lượng tử.
- `maxpool1D.sv`
  - Thực hiện lấy giá trị lớn nhất cho phép toán maxpool.

### 7. Hỗ trợ và tiện ích
- `padding.sv`
  - Tạo offset pad và vị trí input hợp lệ.
- `loop_counter.sv`, `line_buffer1d.sv`, `window_gen.sv`
  - Các khối hỗ trợ vòng lặp, đệm dòng và sinh cửa sổ dữ liệu.
- `residual.sv`
  - Hỗ trợ cộng dư nếu cần mở rộng kiến trúc.
- `write_back.sv`
  - Ghi giá trị tính toán trở lại bộ nhớ OFM.

## Giao tiếp và cấu hình
### Giao diện chính của `cnn1D_core_top`
- Tín hiệu điều khiển chế độ:
  - `start_i`, `busy_o`, `done_o`, `error_o`, `state_dbg_o`
- Cấu hình lớp qua `desc_*`:
  - `desc_op_i`, `desc_ifm_bank_i`, `desc_ofm_bank_i`, `desc_relu_en_i`, `desc_cin_i`, `desc_cout_i`, `desc_len_in_i`, `desc_len_out_i`, `desc_kernel_i`, `desc_stride_i`, `desc_dilation_i`, `desc_pad_left_i`, `desc_ifm_base_i`, `desc_ofm_base_i`, `desc_wgt_base_i`, `desc_param_base_i`
- Giao diện host:
  - `host_we_i`, `host_re_i`, `host_sel_i`, `host_addr_i`, `host_wdata_i`, `host_rdata_o`

### Luồng dữ liệu chung
1. Host nạp IFM, WGT và tham số vào các bank nhớ tương ứng.
2. Host lập trình thông số lớp thông qua `desc_*` và tín hiệu `desc_we_i`.
3. Bắt đầu phép toán với `start_i`.
4. `cnn1d_controller` điều khiển đọc dữ liệu, tích lũy, xử lý ReLU hoặc maxpool, rồi ghi kết quả vào OFM.
5. Host đọc lại kết quả OFM qua `host_rdata_o`.

## Hướng dẫn chạy dự án
- Mở `CNN_AGAiN.xpr` trong Vivado.
- Tổng hợp và định tuyến bằng các script sinh sẵn trong `CNN_AGAiN.runs/` hoặc qua giao diện Vivado.
- Dùng `runme.bat` / `runme.sh` trong các thư mục `synth_1/` và `impl_1/` để tái tạo quy trình Vivado tự động.

## Ghi chú quan trọng
- Thiết kế dùng dữ liệu 8-bit cho IFM/WGT và accumulator 32-bit.
- Bao gồm các phép toán convolution, depthwise convolution, và maxpool trong cùng một kiến trúc.
- Sử dụng kiến trúc ping-pong memory để tránh xung đột giữa host và lõi xử lý.
- Có thể mở rộng thêm các lớp residual hoặc các công nghệ tối ưu hóa khác.

## Tệp chính cần tham khảo
- `cnn1D_core_top.sv`
- `cnn1d_controller.sv`
- `layer_desc.sv`
- `bank_loader.sv`
- `ifm_pingpong_bank.sv`
- `ofm_pingpong_bank.sv`
- `weight_sram_bank.sv`
- `bias_scale_bank.sv`
- `pe_array.sv`
- `requantize.sv`
- `activation_relu.sv`
- `maxpool1D.sv`
- `write_back.sv`

