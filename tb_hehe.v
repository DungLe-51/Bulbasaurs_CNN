`timescale 1ns / 1ps

module tb_hehe;

    // -----------------------------------------------------------
    // 1. CẤU HÌNH THÔNG SỐ (Thay đổi theo đúng NMode của MATLAB)
    // -----------------------------------------------------------
    parameter N_MODE = 258; // Sửa số này thành NMode mà MATLAB báo 
    parameter MAT_SIZE = N_MODE * N_MODE;

    // -----------------------------------------------------------
    // 2. KHAI BÁO TÍN HIỆU KẾT NỐI VỚI MODULE 'hehe'
    // -----------------------------------------------------------
    reg clk;
    reg rst_n;
    reg clr;
    reg en;
    
    reg signed [15:0] T_real, T_imag;
    reg signed [15:0] S_real, S_imag;
    
    wire signed [31:0] Y_real, Y_imag;

    // -----------------------------------------------------------
    // 3. GỌI MODULE 
    // -----------------------------------------------------------
    hehe uut (
        .clk(clk),
        .rst_n(rst_n),
        .clr(clr),
        .en(en),
        .T_real(T_real),
        .T_imag(T_imag),
        .S_real(S_real),
        .S_imag(S_imag),
        .Y_real(Y_real),
        .Y_imag(Y_imag)
    );

    // -----------------------------------------------------------
    // 4. BỘ NHỚ CHỨA FILE .MEM TỪ MATLAB
    // -----------------------------------------------------------
    reg signed [15:0] rom_Tr [0:MAT_SIZE-1];
    reg signed [15:0] rom_Ti [0:MAT_SIZE-1];
    reg signed [15:0] ram_Sr [0:N_MODE-1];
    reg signed [15:0] ram_Si [0:N_MODE-1];

    // -----------------------------------------------------------
    // 5. TẠO XUNG NHỊP CLOCK (Chu kỳ 10ns = 100MHz)
    // -----------------------------------------------------------
    initial clk = 0;
    always #5 clk = ~clk; 

    // Biến chạy vòng lặp và ghi file
    integer row, col;
    integer file_out_r, file_out_i;

    // -----------------------------------------------------------
    // 6. KỊCH BẢN CHẠY CHÍNH (MAIN PROCESS)
    // -----------------------------------------------------------
    initial begin
        // A. Đọc 4 file .mem từ MATLAB
		// đường dẫn tuyệt đối với TÊN FILE CHÍNH XÁC 100%
        $readmemh("C:/Users/LENOVO/Desktop/CNN_QUA/T_inv_real.mem", rom_Tr);
        $readmemh("C:/Users/LENOVO/Desktop/CNN_QUA/T_inv_imag.mem", rom_Ti);
        $readmemh("C:/Users/LENOVO/Desktop/CNN_QUA/Speckle_real.mem", ram_Sr);
        $readmemh("C:/Users/LENOVO/Desktop/CNN_QUA/Speckle_imag.mem", ram_Si);

        // Tạo file để lưu kết quả FPGA tính được
        file_out_r = $fopen("fpga_out_real.txt", "w");
        file_out_i = $fopen("fpga_out_imag.txt", "w");

        // B. Khởi tạo tín hiệu (Reset mạch)
        rst_n = 0; clr = 0; en = 0;
        T_real = 0; T_imag = 0; S_real = 0; S_imag = 0;
        #20; // Đợi 20ns
        rst_n = 1; // Nhả reset để mạch hoạt động
        #10;

        $display("Bat dau tinh toan nhan ma tran...");

        // C. Quét từng hàng của Ma trận T_inv
        for (row = 0; row < N_MODE; row = row + 1) begin
            
            // Nháy cờ 'clr' để xóa bộ cộng dồn cũ
            clr = 1; en = 0;
            #10;
            clr = 0; en = 1; // Bật cho phép cộng dồn
            
            // Quét từng cột (Nhân T_inv với Speckle)
            for (col = 0; col < N_MODE; col = col + 1) begin
                // Bơm dữ liệu vào mạch 'hehe'
                T_real = rom_Tr[row * N_MODE + col];
                T_imag = rom_Ti[row * N_MODE + col];
                S_real = ram_Sr[col];
                S_imag = ram_Si[col];
                
                #10; // Chờ 1 nhịp clock để khối MAC tính và cộng dồn
            end
            
            // Dừng cộng dồn để chốt kết quả của hàng này
            en = 0;
            #10; 
            
            // D. Ghi kết quả Pixel khôi phục ra file text
            $fdisplay(file_out_r, "%d", Y_real);
            $fdisplay(file_out_i, "%d", Y_imag);
            
            // In tiến độ ra màn hình 
            if (row % 50 == 0) $display("Da khoi phuc xong hang %d...", row);
        end

        // E. Kết thúc mô phỏng
        $display("HOAN THANH! Da luu ket qua ra file fpga_out_real/imag.txt");
        $fclose(file_out_r);
        $fclose(file_out_i);
        $finish;
    end

endmodule
