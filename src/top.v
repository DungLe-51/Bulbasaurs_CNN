module CNN_top#(
    parameter IMG_SIZE = 30, IMG_FLAT = IMG_SIZE*IMG_SIZE, // chắc là 30x30 :)))
    parameter DATA_W = 16,
    parameter NUM_F1 = 8, NUM_F2 = 16,
    parameter CONV1_OUT = IMG_SIZE, POOL1_OUT = IMG_SIZE/2,
    parameter CONV2_OUT = POOL1_OUT, POOL2_OUT = CONV2_OUT/2,
    parameter FLAT = POOL2_OUT*POOL2_OUT*NUM_F2, FC_HIDDEN = 128
)(
    input clk,reset,enable;

);
endmodule   