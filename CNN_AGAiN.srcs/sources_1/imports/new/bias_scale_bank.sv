`timescale 1ns/1ps

module bias_scale_bank #(
    parameter int ADDR_W = 8,
    parameter int DEPTH  = 256
)(
    input  logic clk,
    input  logic rst_n,

    input  logic              host_we_i,
    input  logic              host_re_i,
    input  logic [1:0]        host_param_sel_i,
    input  logic [ADDR_W-1:0] host_addr_i,
    input  logic [31:0]       host_wdata_i,
    output logic [31:0]       host_rdata_o,

    input  logic              core_re_i,
    input  logic [ADDR_W-1:0] core_addr_i,
    output logic signed [31:0] bias_o,
    output logic signed [31:0] mult_o,
    output logic [31:0]        shift_o
);
    logic bias_host_en;
    logic mult_host_en;
    logic shift_host_en;

    logic [31:0] bias_host_rdata;
    logic [31:0] mult_host_rdata;
    logic [31:0] shift_host_rdata;

    logic [31:0] bias_core_rdata;
    logic [31:0] mult_core_rdata;
    logic [31:0] shift_core_rdata;

    logic [1:0] host_param_sel_q;

    assign bias_host_en  = (host_we_i || host_re_i) && (host_param_sel_i == 2'd0);
    assign mult_host_en  = (host_we_i || host_re_i) && (host_param_sel_i == 2'd1);
    assign shift_host_en = (host_we_i || host_re_i) && (host_param_sel_i == 2'd2);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            host_param_sel_q <= 2'd0;
        end else begin
            if (host_re_i) begin
                host_param_sel_q <= host_param_sel_i;
            end
        end
    end

    tdp_bram_1clk #(
        .DATA_W(32),
        .ADDR_W(ADDR_W),
        .DEPTH (DEPTH)
    ) u_bias_bram (
        .clk      (clk),

        .a_en_i   (bias_host_en),
        .a_we_i   (host_we_i && (host_param_sel_i == 2'd0)),
        .a_addr_i (host_addr_i),
        .a_wdata_i(host_wdata_i),
        .a_rdata_o(bias_host_rdata),

        .b_en_i   (core_re_i),
        .b_we_i   (1'b0),
        .b_addr_i (core_addr_i),
        .b_wdata_i('0),
        .b_rdata_o(bias_core_rdata)
    );

    tdp_bram_1clk #(
        .DATA_W(32),
        .ADDR_W(ADDR_W),
        .DEPTH (DEPTH)
    ) u_mult_bram (
        .clk      (clk),

        .a_en_i   (mult_host_en),
        .a_we_i   (host_we_i && (host_param_sel_i == 2'd1)),
        .a_addr_i (host_addr_i),
        .a_wdata_i(host_wdata_i),
        .a_rdata_o(mult_host_rdata),

        .b_en_i   (core_re_i),
        .b_we_i   (1'b0),
        .b_addr_i (core_addr_i),
        .b_wdata_i('0),
        .b_rdata_o(mult_core_rdata)
    );

    tdp_bram_1clk #(
        .DATA_W(32),
        .ADDR_W(ADDR_W),
        .DEPTH (DEPTH)
    ) u_shift_bram (
        .clk      (clk),

        .a_en_i   (shift_host_en),
        .a_we_i   (host_we_i && (host_param_sel_i == 2'd2)),
        .a_addr_i (host_addr_i),
        .a_wdata_i(host_wdata_i),
        .a_rdata_o(shift_host_rdata),

        .b_en_i   (core_re_i),
        .b_we_i   (1'b0),
        .b_addr_i (core_addr_i),
        .b_wdata_i('0),
        .b_rdata_o(shift_core_rdata)
    );

    always_comb begin
        unique case (host_param_sel_q)
            2'd0: host_rdata_o = bias_host_rdata;
            2'd1: host_rdata_o = mult_host_rdata;
            2'd2: host_rdata_o = shift_host_rdata;
            default: host_rdata_o = 32'h0;
        endcase

        bias_o  = $signed(bias_core_rdata);
        mult_o  = $signed(mult_core_rdata);
        shift_o = shift_core_rdata;
    end
endmodule