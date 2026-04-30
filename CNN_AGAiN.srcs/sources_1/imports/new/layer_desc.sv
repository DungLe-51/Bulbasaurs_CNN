`timescale 1ns/1ps

module layer_desc_regs #(
    parameter int ADDR_W = 16,
    parameter int LEN_W  = 16,
    parameter int CH_W   = 16,
    parameter int K_W    = 8
)(
    input  logic clk,
    input  logic rst_n,
    input  logic desc_we_i,

    input  logic [1:0]        op_i,
    input  logic              ifm_bank_i,
    input  logic              ofm_bank_i,
    input  logic              relu_en_i,
    input  logic [CH_W-1:0]   cin_i,
    input  logic [CH_W-1:0]   cout_i,
    input  logic [LEN_W-1:0]  len_in_i,
    input  logic [LEN_W-1:0]  len_out_i,
    input  logic [K_W-1:0]    kernel_i,
    input  logic [K_W-1:0]    stride_i,
    input  logic [K_W-1:0]    dilation_i,
    input  logic [K_W-1:0]    pad_left_i,
    input  logic [ADDR_W-1:0] ifm_base_i,
    input  logic [ADDR_W-1:0] ofm_base_i,
    input  logic [ADDR_W-1:0] wgt_base_i,
    input  logic [ADDR_W-1:0] param_base_i,

    output logic [1:0]        op_o,
    output logic              ifm_bank_o,
    output logic              ofm_bank_o,
    output logic              relu_en_o,
    output logic [CH_W-1:0]   cin_o,
    output logic [CH_W-1:0]   cout_o,
    output logic [LEN_W-1:0]  len_in_o,
    output logic [LEN_W-1:0]  len_out_o,
    output logic [K_W-1:0]    kernel_o,
    output logic [K_W-1:0]    stride_o,
    output logic [K_W-1:0]    dilation_o,
    output logic [K_W-1:0]    pad_left_o,
    output logic [ADDR_W-1:0] ifm_base_o,
    output logic [ADDR_W-1:0] ofm_base_o,
    output logic [ADDR_W-1:0] wgt_base_o,
    output logic [ADDR_W-1:0] param_base_o
);
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            op_o         <= 2'd0;
            ifm_bank_o   <= 1'b0;
            ofm_bank_o   <= 1'b1;
            relu_en_o    <= 1'b0;
            cin_o        <= '0;
            cout_o       <= '0;
            len_in_o     <= '0;
            len_out_o    <= '0;
            kernel_o     <= '0;
            stride_o     <= 8'd1;
            dilation_o   <= 8'd1;
            pad_left_o   <= '0;
            ifm_base_o   <= '0;
            ofm_base_o   <= '0;
            wgt_base_o   <= '0;
            param_base_o <= '0;
        end else if (desc_we_i) begin
            op_o         <= op_i;
            ifm_bank_o   <= ifm_bank_i;
            ofm_bank_o   <= ofm_bank_i;
            relu_en_o    <= relu_en_i;
            cin_o        <= cin_i;
            cout_o       <= cout_i;
            len_in_o     <= len_in_i;
            len_out_o    <= len_out_i;
            kernel_o     <= kernel_i;
            stride_o     <= stride_i;
            dilation_o   <= dilation_i;
            pad_left_o   <= pad_left_i;
            ifm_base_o   <= ifm_base_i;
            ofm_base_o   <= ofm_base_i;
            wgt_base_o   <= wgt_base_i;
            param_base_o <= param_base_i;
        end
    end
endmodule