`timescale 1ns / 1ps

module hehe (
    input  wire                 clk,      // Xung nhịp hệ thống
    input  wire                 rst_n,    // Reset tích cực mức thấp (0 là reset)
    input  wire                 clr,      // Tín hiệu xóa bộ cộng dồn (khi bắt đầu tính hàng mới)
    input  wire                 en,       // Tín hiệu cho phép (Enable) tính toán
    
    // Dữ liệu đầu vào 16-bit có dấu (Đọc từ ROM/RAM)
    input  wire signed [15:0]   T_real,   // Trọng số phần thực (T_inv)
    input  wire signed [15:0]   T_imag,   // Trọng số phần ảo (T_inv)
    input  wire signed [15:0]   S_real,   // Speckle phần thực
    input  wire signed [15:0]   S_imag,   // Speckle phần ảo
    
    // Kết quả đầu ra 32-bit có dấu (Chống tràn số khi cộng dồn)
    output reg  signed [31:0]   Y_real,   // Pixel khôi phục phần thực
    output reg  signed [31:0]   Y_imag    // Pixel khôi phục phần ảo
);

    // -----------------------------------------------------------
    // BƯỚC 1: CÁC BỘ NHÂN (Multipliers) - 16 bit x 16 bit = 32 bit
    // Hệ thống sẽ tự động map các phép nhân này vào khối DSP trên FPGA
    // -----------------------------------------------------------
    wire signed [31:0] mult_rr = T_real * S_real;
    wire signed [31:0] mult_ii = T_imag * S_imag;
    wire signed [31:0] mult_ri = T_real * S_imag;
    wire signed [31:0] mult_ir = T_imag * S_real;

    // -----------------------------------------------------------
    // BƯỚC 2: TỔNG HỢP SỐ PHỨC (Cộng / Trừ chéo)
    // -----------------------------------------------------------
    wire signed [31:0] partial_real = mult_rr - mult_ii;
    wire signed [31:0] partial_imag = mult_ri + mult_ir;

    // -----------------------------------------------------------
    // BƯỚC 3: THANH GHI CỘNG DỒN (Accumulator)
    // -----------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            Y_real <= 32'd0;
            Y_imag <= 32'd0;
        end 
        else if (clr) begin
            // Bắt đầu tính 1 pixel khôi phục mới -> Xóa kết quả cũ
            Y_real <= 32'd0;
            Y_imag <= 32'd0;
        end 
        else if (en) begin
            // Đang tính toán -> Cộng dồn giá trị mới vào giá trị cũ
            Y_real <= Y_real + partial_real;
            Y_imag <= Y_imag + partial_imag;
        end
    end

endmodule
