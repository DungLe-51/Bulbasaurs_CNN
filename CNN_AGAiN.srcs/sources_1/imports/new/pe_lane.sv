`timescale 1ns/1ps

module pe_lane #(
    parameter int DATA_W = 8,
    parameter int WGT_W  = 8,
    parameter int ACC_W  = 32
)(
    input  logic signed [DATA_W-1:0] act_i,
    input  logic signed [WGT_W-1:0]  wgt_i,
    output logic signed [ACC_W-1:0]  prod_o
);
    mac_int8 #(
        .DATA_W(DATA_W),
        .WGT_W (WGT_W),
        .ACC_W (ACC_W)
    ) u_mac_int8 (
        .act_i (act_i),
        .wgt_i (wgt_i),
        .prod_o(prod_o)
    );
endmodule