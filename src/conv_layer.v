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
            done <= 0;
        end else begin
            state <= next_state; 
            case (state)
            IDLE: if (enable) begin
                state <= 1;
                f <= 0; y <= 0; x <= 0; k <= 0;
                done <= 0;
            end
            if (state == CONV) begin
                // thực hiện convolution
                for (f = 0; f < NUM_F1; f = f + 1) begin
                    for (y = 0; y < IMG_SIZE-2; y = y + 1) begin
                        for (x = 0; x < IMG_SIZE-2; x = x + 1) begin
                            accum = bias[f];
                            for (k = 0; k < 9; k = k + 1) begin
                                accum = accum + data_in[(y+(k/3))*IMG_SIZE + (x+(k%3))] * kernel[f][k];
                            end
                            conv_out[f*IMG_FLAT + y*IMG_SIZE + x] <= accum[DATA_WIDTH+8:8]; // lấy phần có dấu và cắt về DATA_WIDTH
                        end
                    end
                end
                done <= 1;
            end else begin
                done <= 0;
            end
        end
    end
endmodule