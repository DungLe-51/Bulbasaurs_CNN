`timescale 1ns/1ps

module weight_sram_bank #(
    parameter int DATA_W = 8,
    parameter int ADDR_W = 12,
    parameter int DEPTH  = 4096
)(
    input  logic clk,
    input  logic rst_n,

    input  logic                     host_we_i,
    input  logic                     host_re_i,
    input  logic [ADDR_W-1:0]        host_addr_i,
    input  logic [31:0]              host_wdata_i,
    output logic [31:0]              host_rdata_o,

    input  logic                     core_re_i,
    input  logic [ADDR_W-1:0]        core_addr_i,
    output logic signed [DATA_W-1:0] core_rdata_o
);
    logic [DATA_W-1:0] host_rdata;
    logic [DATA_W-1:0] core_rdata;

    tdp_bram_1clk #(
        .DATA_W(DATA_W),
        .ADDR_W(ADDR_W),
        .DEPTH (DEPTH)
    ) u_weight_bram (
        .clk      (clk),

        .a_en_i   (host_we_i || host_re_i),
        .a_we_i   (host_we_i),
        .a_addr_i (host_addr_i),
        .a_wdata_i(host_wdata_i[DATA_W-1:0]),
        .a_rdata_o(host_rdata),

        .b_en_i   (core_re_i),
        .b_we_i   (1'b0),
        .b_addr_i (core_addr_i),
        .b_wdata_i('0),
        .b_rdata_o(core_rdata)
    );

    always_comb begin
        host_rdata_o = {{(32-DATA_W){host_rdata[DATA_W-1]}}, host_rdata};
        core_rdata_o = $signed(core_rdata);
    end
endmodule