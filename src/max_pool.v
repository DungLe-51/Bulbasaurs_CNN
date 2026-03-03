module max_pooling_layer #(
    parameter IMG_SIZE = 30, IMG_FLAT = IMG_SIZE*IMG_SIZE, // chắc là 30x30 :)))
    parameter DATA_WIDTH = 16,// đây là độ lệch
    parameter NUM_F1 = 8, NUM_F2 = 16,// mấy cái này là các thành phần cần phải có
    parameter CONV1_OUT = IMG_SIZE, POOL1_OUT = IMG_SIZE/2,
    parameter CONV2_OUT = POOL1_OUT, POOL2_OUT = CONV2_OUT/2,
    parameter FLAT = POOL2_OUT*POOL2_OUT*NUM_F2, FC_HIDDEN = 128
) (
    input  wire clk,
    input  wire reset,
    input  wire start,
    input  signed [DATA_WIDTH-1:0] data_in [0:IMG_SIZE*IMG_SIZE*NUM_F1-1],
    output reg  signed [DATA_WIDTH-1:0] pool_out [0:(IMG_SIZE/2)*(IMG_SIZE/2)*NUM_F1-1],
    output reg  done
);

    localparam IDLE = 1'b0, POOL = 1'b1;
    reg state, next_state;
    integer c, y, x;

    always @(posedge clk or posedge reset) begin
        if (reset) state <= IDLE;
        else       state <= next_state;
    end

    always @(*) begin
        next_state = state;
        done = 0;

        case (state)
            IDLE: if (start) next_state = POOL;

            POOL: begin
                for (c = 0; c < NUM_F1; c = c + 1) begin
                    for (y = 0; y < IMG_SIZE/2; y = y + 1) begin
                        for (x = 0; x < IMG_SIZE/2; x = x + 1) begin
                            signed [DATA_WIDTH-1:0] a = data_in[c*IMG_SIZE*IMG_SIZE + (y*2)*IMG_SIZE   + x*2];
                            signed [DATA_WIDTH-1:0] b = data_in[c*IMG_SIZE*IMG_SIZE + (y*2)*IMG_SIZE   + x*2+1];
                            signed [DATA_WIDTH-1:0] cc= data_in[c*IMG_SIZE*IMG_SIZE + (y*2+1)*IMG_SIZE + x*2];
                            signed [DATA_WIDTH-1:0] d = data_in[c*IMG_SIZE*IMG_SIZE + (y*2+1)*IMG_SIZE + x*2+1];
                            
                            signed [DATA_WIDTH-1:0] max1 = (a > b) ? a : b;
                            signed [DATA_WIDTH-1:0] max2 = (cc > d) ? cc : d;
                            pool_out[c*(IMG_SIZE/2)*(IMG_SIZE/2) + y*(IMG_SIZE/2) + x] = (max1 > max2) ? max1 : max2;
                        end
                    end
                end
                next_state = IDLE;
                done = 1;
            end
        endcase
    end
endmodule