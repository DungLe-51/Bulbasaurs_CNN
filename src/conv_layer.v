module conv_layer #(
    parameter IMG_SIZE = 30, IMG_FLAT = IMG_SIZE*IMG_SIZE, // chắc là 30x30 :)))
    parameter DATA_WIDTH = 16,// đây là độ lệch
    parameter NUM_F1 = 8, NUM_F2 = 16,// mấy cái này là các thành phần cần phải có
    parameter CONV1_OUT = IMG_SIZE, POOL1_OUT = IMG_SIZE/2,
    parameter CONV2_OUT = POOL1_OUT, POOL2_OUT = CONV2_OUT/2,
    parameter FLAT = POOL2_OUT*POOL2_OUT*NUM_F2, FC_HIDDEN = 128
)(
     input clk,reset,enable,
     input signed [DATA_WIDTH-1:0] data_in [0:IMG_SIZE*IMG_SIZE-1],
     output signed [DATA_WIDTH-1:0] conv_out[0:(IMG_SIZE)*(IMG_SIZE)*NUM_F1-1],
     output reg done,
);
    reg signed [DATA_WIDTH-1:0] kernel [0:NUM_FILTERS-1][0:8];   
    reg signed [DATA_WIDTH-1:0] bias   [0:NUM_FILTERS-1];
    initial begin
        //tí bổ sung các trọng số để convert sau
        // t đang tính làm 1 cái cylinde array giống kiểu tụi hàn làm 
        // nó nhanh mà đỡ tốn clock
    end

    localparam IDLE =1'b0,CONV=1'b1 ;
    reg state,next_state;
    integer f, y, x, k;
    reg signed [DATA_WIDTH+8:0] accum;   // đủ rộng để tránh tràn
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            state <= IDLE;
            done  <= 1'b0;
            f <= 0; y <= 0; x <= 0; k <= 0;
            accum <= 0;
        end else begin
            state <= next_state;
        end
    end
    always @(*) begin
        next_state = state;
        done       = 1'b0;

        case (state)
            IDLE: begin
                if (enable) begin
                    next_state = CONV;
                    f = 0; y = 0; x = 0; k = 0;
                end
            end

            CONV: begin
                // Tính 1 kernel element mỗi clock
                if (k == 0) 
                    accum = bias[f];                    // khởi tạo bias
                else 
                    accum = accum;                      // giữ giá trị cũ

                // Tính convolution
                integer ky = k / 3;
                integer kx = k % 3;
                
                accum = accum + kernel[f][k] * data_in[(y + ky)*IMG_SIZE + (x + kx)];

                // Khi tính xong 9 phần tử kernel → lưu kết quả
                if (k == 8) begin
                    conv_out[f*IMG_SIZE*IMG_SIZE + y*IMG_SIZE + x] = accum[DATA_WIDTH+3:4]; // Q8.8

                    // Tăng tọa độ
                    k = 0;
                    x = x + 1;
                    if (x == IMG_SIZE) begin
                        x = 0;
                        y = y + 1;
                        if (y == IMG_SIZE) begin
                            y = 0;
                            f = f + 1;
                            if (f == NUM_F1) begin
                                next_state = IDLE;
                                done = 1'b1;
                            end
                        end
                    end
                end else begin
                    k = k + 1;
                end
            end

            default: next_state = IDLE;
        endcase
    end
endmodule