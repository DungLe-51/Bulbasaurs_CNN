`timescale 1ns/1ps

module ofm_pingpong_bank #(
    parameter int DATA_W = 8,
    parameter int ADDR_W = 12,
    parameter int DEPTH  = 4096
)(
    input  logic clk,
    input  logic rst_n,

    input  logic                     host_we_i,
    input  logic                     host_re_i,
    input  logic                     host_bank_i,
    input  logic [ADDR_W-1:0]        host_addr_i,
    input  logic [31:0]              host_wdata_i,
    output logic [31:0]              host_rdata_o,

    input  logic                     core_we_i,
    input  logic                     core_re_i,
    input  logic                     core_bank_i,
    input  logic [ADDR_W-1:0]        core_addr_i,
    input  logic signed [DATA_W-1:0] core_wdata_i,
    output logic signed [DATA_W-1:0] core_rdata_o
);
    logic ping_host_en;
    logic pong_host_en;
    logic ping_core_en;
    logic pong_core_en;

    logic [DATA_W-1:0] ping_host_rdata;
    logic [DATA_W-1:0] pong_host_rdata;
    logic [DATA_W-1:0] ping_core_rdata;
    logic [DATA_W-1:0] pong_core_rdata;

    logic host_bank_q;
    logic core_bank_q;

    assign ping_host_en = (host_we_i || host_re_i) && (host_bank_i == 1'b0);
    assign pong_host_en = (host_we_i || host_re_i) && (host_bank_i == 1'b1);

    assign ping_core_en = (core_we_i || core_re_i) && (core_bank_i == 1'b0);
    assign pong_core_en = (core_we_i || core_re_i) && (core_bank_i == 1'b1);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            host_bank_q <= 1'b0;
            core_bank_q <= 1'b0;
        end else begin
            if (host_re_i) begin
                host_bank_q <= host_bank_i;
            end
            if (core_re_i) begin
                core_bank_q <= core_bank_i;
            end
        end
    end

    tdp_bram_1clk #(
        .DATA_W(DATA_W),
        .ADDR_W(ADDR_W),
        .DEPTH (DEPTH)
    ) u_ping_bram (
        .clk      (clk),

        .a_en_i   (ping_host_en),
        .a_we_i   (host_we_i && (host_bank_i == 1'b0)),
        .a_addr_i (host_addr_i),
        .a_wdata_i(host_wdata_i[DATA_W-1:0]),
        .a_rdata_o(ping_host_rdata),

        .b_en_i   (ping_core_en),
        .b_we_i   (core_we_i && (core_bank_i == 1'b0)),
        .b_addr_i (core_addr_i),
        .b_wdata_i(core_wdata_i),
        .b_rdata_o(ping_core_rdata)
    );

    tdp_bram_1clk #(
        .DATA_W(DATA_W),
        .ADDR_W(ADDR_W),
        .DEPTH (DEPTH)
    ) u_pong_bram (
        .clk      (clk),

        .a_en_i   (pong_host_en),
        .a_we_i   (host_we_i && (host_bank_i == 1'b1)),
        .a_addr_i (host_addr_i),
        .a_wdata_i(host_wdata_i[DATA_W-1:0]),
        .a_rdata_o(pong_host_rdata),

        .b_en_i   (pong_core_en),
        .b_we_i   (core_we_i && (core_bank_i == 1'b1)),
        .b_addr_i (core_addr_i),
        .b_wdata_i(core_wdata_i),
        .b_rdata_o(pong_core_rdata)
    );

    always_comb begin
        if (host_bank_q) begin
            host_rdata_o = {{(32-DATA_W){pong_host_rdata[DATA_W-1]}}, pong_host_rdata};
        end else begin
            host_rdata_o = {{(32-DATA_W){ping_host_rdata[DATA_W-1]}}, ping_host_rdata};
        end

        if (core_bank_q) begin
            core_rdata_o = $signed(pong_core_rdata);
        end else begin
            core_rdata_o = $signed(ping_core_rdata);
        end
    end
endmodule