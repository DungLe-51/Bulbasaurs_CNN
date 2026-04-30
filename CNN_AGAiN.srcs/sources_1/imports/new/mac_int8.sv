`timescale 1ns/1ps

module mac_int8 #(
    parameter int DATA_W = 8,
    parameter int WGT_W  = 8,
    parameter int ACC_W  = 32
)(
    input  logic signed [DATA_W-1:0] act_i,
    input  logic signed [WGT_W-1:0]  wgt_i,
    output logic signed [ACC_W-1:0]  prod_o
);
    logic signed [DATA_W+WGT_W-1:0] prod_s;

    assign prod_s = act_i * wgt_i;
    assign prod_o = {{(ACC_W-(DATA_W+WGT_W)){prod_s[DATA_W+WGT_W-1]}}, prod_s};
endmodule