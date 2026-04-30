`timescale 1ns/1ps

module weight_addr_gen #(
    parameter int ADDR_W = 16,
    parameter int CH_W   = 16,
    parameter int K_W    = 8
)(
    input  logic [1:0]        op_i,
    input  logic [ADDR_W-1:0] base_i,
    input  logic [CH_W-1:0]   oc_i,
    input  logic [CH_W-1:0]   ic_i,
    input  logic [CH_W-1:0]   cin_i,
    input  logic [K_W-1:0]    k_i,
    input  logic [K_W-1:0]    kernel_i,
    output logic [ADDR_W-1:0] addr_o
);
    localparam logic [1:0] OP_CONV1D   = 2'd0;
    localparam logic [1:0] OP_DWCONV1D = 2'd1;

    logic [ADDR_W+CH_W+K_W+8-1:0] addr_long;

    always_comb begin
        if (op_i == OP_DWCONV1D) begin
            addr_long = base_i + (oc_i * kernel_i) + k_i;
        end else begin
            addr_long = base_i + (((oc_i * cin_i) + ic_i) * kernel_i) + k_i;
        end
        addr_o = addr_long[ADDR_W-1:0];
    end
endmodule