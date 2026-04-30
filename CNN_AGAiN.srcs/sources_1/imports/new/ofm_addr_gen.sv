`timescale 1ns/1ps

module ofm_addr_gen_1d #(
    parameter int ADDR_W = 16,
    parameter int LEN_W  = 16,
    parameter int CH_W   = 16
)(
    input  logic [ADDR_W-1:0] base_i,
    input  logic [CH_W-1:0]   channel_i,
    input  logic [LEN_W-1:0]  pos_i,
    input  logic [LEN_W-1:0]  len_i,
    output logic [ADDR_W-1:0] addr_o
);
    logic [ADDR_W+LEN_W+CH_W-1:0] addr_long;

    always_comb begin
        addr_long = base_i + (channel_i * len_i) + pos_i;
        addr_o = addr_long[ADDR_W-1:0];
    end
endmodule