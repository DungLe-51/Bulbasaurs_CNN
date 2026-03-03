module fc_layer #(
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
    input  signed [DATA_WIDTH-1:0] data_in [0:FLAT-1],
    output reg  signed [DATA_WIDTH-1:0] fc_out [0:FC_HIDDEN-1],
    output reg  done
);

    reg signed [DATA_WIDTH-1:0] weight [0:FC_HIDDEN*FLAT-1];
    reg signed [DATA_WIDTH-1:0] bias   [0:FC_HIDDEN-1];

    initial begin
        // Tí bổ sung trọng số và bias cho fully connected layer
        // Có thể load từ file hoặc hardcode tùy ý
    end

    localparam IDLE = 1'b0, COMPUTE = 1'b1;
    reg state, next_state;
    integer j, i;
    reg signed [DATA_WIDTH+10:0] accum;

    always @(posedge clk or posedge reset) begin
        if (reset) state <= IDLE;
        else       state <= next_state;
    end

    always @(*) begin
        next_state = state;
        done = 0;

        case (state)
            IDLE: if (start) next_state = COMPUTE;

            COMPUTE: begin
                for (j = 0; j < FC_HIDDEN; j = j + 1) begin
                    accum = bias[j];
                    for (i = 0; i < FLAT; i = i + 1) begin
                        accum = accum + weight[j*FLAT + i] * data_in[i];// Tính tích chập nha 
                    end
                    fc_out[j] = accum[DATA_WIDTH+7:8];   // Q8.8
                end
                next_state = IDLE;
                done = 1;
            end
        endcase
    end
endmodule