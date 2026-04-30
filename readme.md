# CNN-1D Accelerator Core V1

## 1. Mục tiêu dự án

Dự án này hiện thực một **CNN-1D inference accelerator core** bằng SystemVerilog để chạy trên FPGA Xilinx và có thể tái sử dụng kiến trúc cho hướng ASIC sau này. Core được thiết kế để nhận input activation, weight, bias, quantization parameters đã được train/export từ phần mềm, sau đó thực hiện suy luận CNN-1D theo kiểu fixed-point.

Mục tiêu dài hạn của core là phục vụ bài toán **Distributed Acoustic Sensing (DAS) earthquake phase picking**, trong đó tín hiệu DAS được xử lý theo patch thời gian/kênh và output cuối cùng có thể là các lớp như `P wave`, `S wave`, `noise`.

Core hiện tại là bản **V1 correctness-first**: ưu tiên chạy đúng, synthesize được, implementation được và kiểm chứng được memory inference trước khi mở rộng lên PE array lớn hơn.

---

## 2. Trạng thái hiện tại

Trạng thái hiện tại:

- Đã add đầy đủ RTL vào Vivado.
- Đã sửa lỗi thiếu module `adder_tree`.
- Đã thêm `tdp_bram_1clk.sv` để chuẩn hóa template BRAM.
- Đã thay memory bank cũ bằng memory-wrapper style:
  - `ifm_pingpong_bank.sv`
  - `ofm_pingpong_bank.sv`
  - `weight_sram_bank.sv`
  - `bias_scale_bank.sv`
- Đã synthesis thành công.
- Đã run implementation thành công.
- Cấu hình RAM lớn vẫn pass implementation:

```systemverilog
parameter int ADDR_W       = 16;
parameter int PARAM_ADDR_W = 12;
parameter int IFM_DEPTH    = 65536;
parameter int OFM_DEPTH    = 65536;
parameter int WGT_DEPTH    = 65536;
parameter int PARAM_DEPTH  = 4096;
```

Điều này cho thấy Vivado đã nhận diện memory theo hướng BRAM hợp lệ thay vì dissolve RAM thành LUT/flip-flop.

---

## 3. Kiến trúc tổng quan

```text
cnn1d_core_top
  |
  |-- control
  |     |-- layer_desc_regs
  |     |-- cnn1d_controller
  |
  |-- address
  |     |-- padding_dilation_unit
  |     |-- ifm_addr_gen_1d
  |     |-- ofm_addr_gen_1d
  |     |-- weight_addr_gen
  |
  |-- memory
  |     |-- ifm_pingpong_bank
  |     |     |-- ping_bram : tdp_bram_1clk
  |     |     |-- pong_bram : tdp_bram_1clk
  |     |
  |     |-- ofm_pingpong_bank
  |     |     |-- ping_bram : tdp_bram_1clk
  |     |     |-- pong_bram : tdp_bram_1clk
  |     |
  |     |-- weight_sram_bank
  |     |     |-- weight_bram : tdp_bram_1clk
  |     |
  |     |-- bias_scale_bank
  |     |     |-- bias_bram  : tdp_bram_1clk
  |     |     |-- mult_bram  : tdp_bram_1clk
  |     |     |-- shift_bram : tdp_bram_1clk
  |
  |-- compute
  |     |-- pe_array
  |     |-- pe_lane
  |     |-- mac_int8
  |     |-- adder_tree
  |
  |-- post
        |-- requantize
        |-- activation_relu
        |-- maxpool1d
        |-- writeback_unit
```

---

## 4. Chức năng core V1

Core V1 hỗ trợ các phép toán chính:

| Chức năng | Trạng thái |
|---|---|
| Normal Conv1D | Có |
| Depthwise Conv1D | Có |
| Pointwise Conv1D | Có, dùng `kernel = 1` |
| Padding | Có |
| Stride | Có |
| Dilation / atrous Conv1D | Có |
| Bias add | Có |
| Requantization | Có |
| ReLU | Có |
| MaxPool1D | Có |
| Ping-pong IFM/OFM bank | Có |
| Weight BRAM bank | Có |
| Bias/mult/shift BRAM bank | Có |
| AXI wrapper | Chưa có |
| DMA | Chưa có |
| Multi-lane PE array lớn | Chưa có |
| Line buffer/window buffer tối ưu | Chưa tích hợp |
| Residual add runtime path | Chưa tích hợp vào top V1 |
| Phase picker postprocess | Chưa có |

---

## 5. Data type

Core đang dùng fixed-point integer inference:

```text
Activation input  : signed int8
Weight            : signed int8
Bias              : signed int32
Accumulator       : signed int32
Quant multiplier  : signed int32
Quant shift       : 5-bit value stored in 32-bit word
Output activation : signed int8
```

Công thức tính Conv1D thường:

```text
O[oc, t] = bias[oc]
         + sum_ic sum_k I[ic, t * stride + k * dilation - pad_left]
                      * W[oc, ic, k]
```

Công thức Depthwise Conv1D:

```text
O[c, t] = bias[c]
        + sum_k I[c, t * stride + k * dilation - pad_left]
                * W[c, k]
```

Requantization:

```text
y_int8 = saturate_int8((acc_int32 * mult_int32[oc]) >>> shift[oc])
```

ReLU optional:

```text
if relu_en:
    y = max(y, 0)
```

---

## 6. Memory map

Host memory select:

| `host_sel_i` | Bank |
|---:|---|
| 0 | IFM ping |
| 1 | IFM pong |
| 2 | Weight bank |
| 3 | Bias bank |
| 4 | Quant multiplier bank |
| 5 | Quant shift bank |
| 6 | OFM ping |
| 7 | OFM pong |

Tensor layout:

```text
IFM[ic][t]
addr = ifm_base + ic * len_in + t

OFM[oc][t]
addr = ofm_base + oc * len_out + t

Normal Conv weight W[oc][ic][k]
addr = wgt_base + ((oc * cin + ic) * kernel + k)

Depthwise Conv weight W[c][k]
addr = wgt_base + c * kernel + k

BIAS[oc]
addr = param_base + oc

MULT[oc]
addr = param_base + oc

SHIFT[oc]
addr = param_base + oc
```

---

## 7. FSM của controller

FSM chính trong `cnn1d_controller.sv`:

```text
ST_IDLE
  Chờ start_i.

ST_VALIDATE
  Kiểm tra cấu hình layer: op, cin, cout, len, kernel, stride, dilation.

ST_BIAS_READ
  Phát read enable tới bias/mult/shift bank.

ST_BIAS_WAIT
  Chờ latency 1 chu kỳ của BRAM.

ST_INIT_ACC
  Khởi tạo accumulator bằng bias hoặc giá trị min cho maxpool.

ST_MAC_READ
  Tính địa chỉ IFM/weight, phát lệnh đọc BRAM.

ST_MAC_ACCUM
  Nhận dữ liệu từ BRAM, thực hiện MAC hoặc maxpool update.

ST_POST
  Requantize, optional ReLU, chuẩn bị dữ liệu ghi OFM.

ST_OFM_WRITE
  Ghi output vào OFM bank.

ST_DONE
  Báo done_o = 1.

ST_ERROR
  Báo error_o = 1 nếu descriptor sai.
```

---

## 8. Cấu hình RAM hiện tại và ước lượng BRAM

Với cấu hình lớn:

```systemverilog
ADDR_W       = 16
PARAM_ADDR_W = 12
IFM_DEPTH    = 65536
OFM_DEPTH    = 65536
WGT_DEPTH    = 65536
PARAM_DEPTH  = 4096
```

Kích thước logic memory:

```text
IFM ping  = 65536 x 8  = 524288 bit
IFM pong  = 65536 x 8  = 524288 bit
OFM ping  = 65536 x 8  = 524288 bit
OFM pong  = 65536 x 8  = 524288 bit
Weight    = 65536 x 8  = 524288 bit
Bias      = 4096  x 32 = 131072 bit
Mult      = 4096  x 32 = 131072 bit
Shift     = 4096  x 32 = 131072 bit
```

Tổng raw memory khoảng:

```text
3,014,656 bit ≈ 2.875 Mib ≈ 368 KiB
```

Ước lượng BRAM36 theo cấu hình 4096x9 hoặc 1024x36 thường vào khoảng:

```text
IFM ping/pong : ~32 RAMB36
OFM ping/pong : ~32 RAMB36
Weight        : ~16 RAMB36
Bias/mult/shift: ~12 RAMB36
Total         : ~92 RAMB36, chưa tính overhead của tool
```

Sau implementation cần kiểm tra `report_utilization` để xác nhận RAM đã vào `RAMB18/RAMB36`, không bị biến thành LUT/FF.

---

## 9. Lệnh Vivado nên dùng

Set SystemVerilog:

```tcl
set_property file_type SystemVerilog [get_files *.sv]
update_compile_order -fileset sources_1
```

Reset và chạy lại synthesis/implementation:

```tcl
reset_run synth_1
reset_run impl_1
launch_runs synth_1 -jobs 4
wait_on_run synth_1
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1
```

Xuất report:

```tcl
open_run synth_1
report_utilization -hierarchical -file reports/util_synth_hier.rpt
report_timing_summary -file reports/timing_synth.rpt

open_run impl_1
report_utilization -hierarchical -file reports/util_impl_hier.rpt
report_timing_summary -file reports/timing_impl.rpt
report_power -file reports/power_impl.rpt
```

Kiểm tra BRAM:

```tcl
report_utilization -hierarchical
```

Cần thấy có sử dụng `RAMB18`/`RAMB36`.

---

## 10. Cách chạy testbench

Các testbench cần để trong `Simulation Sources`, không để trong `Design Sources`:

```text
tb_mac_int8.sv
tb_conv1d_single_layer.sv
tb_cnn1d_core.sv
```

Nên chạy theo thứ tự:

```text
1. tb_mac_int8
2. tb_conv1d_single_layer
3. tb_cnn1d_core
```

Mục tiêu test:

- Kiểm tra MAC int8 signed.
- Kiểm tra Conv1D đơn giản.
- Kiểm tra padding/stride/dilation/depthwise/maxpool.
- Kiểm tra output đọc từ OFM bank đúng với golden vector.

---

## 11. FPGA bring-up checklist

Trước khi chạy board thật:

- [ ] Synthesis pass.
- [ ] Implementation pass.
- [ ] Timing met: WNS >= 0.
- [ ] BRAM đã infer đúng: có RAMB18/RAMB36 trong utilization.
- [ ] Không còn critical warning liên quan đến latch, multi-driver, unconstrained clock.
- [ ] Behavioral simulation pass.
- [ ] Post-synthesis functional simulation pass nếu cần.
- [ ] Có golden vectors từ Python int8 reference.
- [ ] Có script export weight/bias/mult/shift từ PyTorch.
- [ ] Có driver hoặc interface nạp memory vào core.

---

## 12. Hướng đi FPGA tiếp theo

### Bước 1: Đóng băng RTL V1

Tag phiên bản hiện tại là:

```text
cnn1d_core_v1_bram_pass_impl
```

Lưu lại:

```text
- RTL source
- Vivado project TCL
- synthesis report
- implementation report
- timing report
- utilization report
- power report
```

### Bước 2: Verify chức năng bằng simulation

Synthesis/implementation pass chưa chứng minh core chạy đúng. Cần chạy testbench và so sánh với Python golden model.

### Bước 3: Viết Python int8 reference model

Python model cần mô phỏng đúng:

```text
Conv1D
Depthwise Conv1D
Pointwise Conv1D
MaxPool1D
Padding
Stride
Dilation
Bias
Requantization
ReLU
Saturation int8
```

Output Python phải bit-exact với RTL.

### Bước 4: Tạo exporter weight/bias/scale

Từ PyTorch:

```text
train float32
fold BatchNorm
quantize int8
export weight_int8.mem
export bias_int32.mem
export mult_int32.mem
export shift.mem
export layer_desc.json hoặc layer_desc.mem
```

### Bước 5: Tạo AXI wrapper

Cho Zynq nên tạo:

```text
AXI4-Lite:
  control/status/descriptor registers

AXI BRAM hoặc AXI DMA:
  load IFM/weight/bias/mult/shift
  read OFM
```

Register tối thiểu:

```text
CTRL       : start, soft_reset
STATUS     : busy, done, error
DESC_OP
DESC_CIN
DESC_COUT
DESC_LEN_IN
DESC_LEN_OUT
DESC_KERNEL
DESC_STRIDE
DESC_DILATION
DESC_PAD_LEFT
DESC_BASE_ADDR
```

### Bước 6: Chạy board thật

Trình tự board test:

```text
1. PS/host ghi IFM vào IFM ping.
2. PS/host ghi weight vào weight bank.
3. PS/host ghi bias/mult/shift.
4. PS/host ghi descriptor.
5. PS/host set start.
6. Chờ done interrupt hoặc polling done.
7. Đọc OFM pong.
8. So sánh với Python golden vector.
```

### Bước 7: Đo performance

Đo:

```text
latency cycles/layer
latency microseconds/layer
throughput MAC/s
BRAM bandwidth utilization
Fmax
power
resource utilization
```

Công thức ước lượng V1 scalar:

```text
Normal Conv1D cycles ≈ COUT * LOUT * (2 * CIN * K + overhead)
Depthwise cycles     ≈ COUT * LOUT * (2 * K + overhead)
```

V1 hiện chỉ có 1 MAC lane nên đúng để verify, nhưng chưa tối ưu tốc độ.

### Bước 8: Nâng lên V2 sau khi V1 bit-exact

V2 nên thêm:

```text
- IC_PAR / OC_PAR nhiều MAC song song
- weight_rf
- bias_scale_rf
- line_buffer_1d
- window_gen_1d
- residual_add runtime path
- tile_scheduler
- double-buffered weight prefetch
```

---

## 13. Hướng đi ASIC sau FPGA

Không dùng Xilinx BRAM IP cho ASIC. Với ASIC, giữ kiến trúc memory-wrapper style:

```text
cnn1d_core_top giữ nguyên
memory bank giữ interface tương tự
bên trong tdp_bram_1clk thay bằng SRAM macro wrapper
```

ASIC memory flow:

```text
1. Chọn SRAM depth/width.
2. Dùng SRAM compiler hoặc PDK SRAM macro.
3. Nhận .lib, .lef, .gds, .v model.
4. Instantiate SRAM macro trong wrapper.
5. Synthesis với standard-cell .lib + SRAM .lib.
6. Place & route với standard-cell LEF + SRAM LEF.
7. Signoff STA, DRC, LVS, IR/EM.
```

Với ASIC final, cân nhắc dùng SRAM 1RW + arbitration thay vì true dual-port nếu muốn giảm area/power.

---

## 14. Hướng đi model cho DAS

DeepSubDAS dùng semantic segmentation cho DAS patch nhiều channel/time và output P/S/noise. Với CNN-1D accelerator này, hướng phù hợp là train một student model CNN-1D/TCN:

```text
Input DAS patch
  -> Temporal Conv1D
  -> Dilated Temporal Conv1D / TCN
  -> Spatial Conv1D theo DAS channel
  -> Pointwise Conv1D
  -> Head 3 lớp: P, S, noise
```

Dữ liệu DeepSubDAS đã chuẩn hóa theo hướng:

```text
band-pass 1-20 Hz
resample 100 Hz
40 s segment
patch 200-500 DAS channels
label P/S/noise
```

Core hiện tại là block tính toán layer. Toàn bộ mạng nhiều layer cần chạy bằng cách nạp descriptor/weight từng layer và ping-pong IFM/OFM.

---

## 15. Quy tắc quan trọng

- Synthesis/implementation pass chưa đủ: phải simulation và bit-exact với Python.
- Không dùng `dissolveMemorySizeLimit` để ép RAM thành FF/LUT.
- Memory lớn phải vào BRAM trên FPGA hoặc SRAM macro trên ASIC.
- Testbench phải nằm trong Simulation Sources.
- `.sv` phải được set là SystemVerilog.
- Giữ memory wrapper tách biệt để dễ chuyển FPGA -> ASIC.
- Không mở rộng PE array trước khi V1 đúng hoàn toàn.

---

## 16. Roadmap đề xuất

```text
M0: RTL V1 BRAM synthesis/implementation pass     [DONE]
M1: Behavioral simulation pass                    [NEXT]
M2: Python int8 golden model bit-exact            [NEXT]
M3: Randomized layer tests                        [NEXT]
M4: AXI4-Lite wrapper + board test                [NEXT]
M5: PyTorch export weight/bias/mult/shift         [NEXT]
M6: Run 1 real CNN layer from trained model        [NEXT]
M7: Run full small CNN-1D model layer-by-layer     [NEXT]
M8: Measure latency/resource/power                [NEXT]
M9: Design V2 parallel PE array                   [LATER]
M10: ASIC SRAM wrapper + synthesis                [LATER]
```

---

## 17. Ghi chú phiên bản

Phiên bản hiện tại nên được xem là:

```text
CNN1D_ACCEL_CORE_V1_BRAM
```

Mục tiêu của V1:

```text
đúng chức năng
synthesize được
implementation được
BRAM inference đúng
chuẩn bị đường đi sang board test và ASIC memory wrapper
```

Mục tiêu của V2:

```text
tăng throughput bằng nhiều MAC lane
reuse input/weight tốt hơn
thêm line buffer/window buffer
thêm weight RF
thêm residual/tile scheduler
phù hợp hơn cho DAS real-time
```