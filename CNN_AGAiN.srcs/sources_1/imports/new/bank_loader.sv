`timescale 1ns/1ps

module bank_loader #(
    parameter int ADDR_W = 16
)(
    input  logic              host_we_i,
    input  logic              host_re_i,
    input  logic [3:0]        host_sel_i,
    input  logic [ADDR_W-1:0] host_addr_i,
    input  logic [31:0]       host_wdata_i,

    output logic              ifm_we_o,
    output logic              ifm_re_o,
    output logic              ifm_bank_o,
    output logic [ADDR_W-1:0] ifm_addr_o,
    output logic [31:0]       ifm_wdata_o,

    output logic              wgt_we_o,
    output logic              wgt_re_o,
    output logic [ADDR_W-1:0] wgt_addr_o,
    output logic [31:0]       wgt_wdata_o,

    output logic              param_we_o,
    output logic              param_re_o,
    output logic [1:0]        param_sel_o,
    output logic [ADDR_W-1:0] param_addr_o,
    output logic [31:0]       param_wdata_o,

    output logic              ofm_we_o,
    output logic              ofm_re_o,
    output logic              ofm_bank_o,
    output logic [ADDR_W-1:0] ofm_addr_o,
    output logic [31:0]       ofm_wdata_o
);
    always_comb begin
        ifm_we_o     = host_we_i && ((host_sel_i == 4'd0) || (host_sel_i == 4'd1));
        ifm_re_o     = host_re_i && ((host_sel_i == 4'd0) || (host_sel_i == 4'd1));
        ifm_bank_o   = (host_sel_i == 4'd1);
        ifm_addr_o   = host_addr_i;
        ifm_wdata_o  = host_wdata_i;

        wgt_we_o     = host_we_i && (host_sel_i == 4'd2);
        wgt_re_o     = host_re_i && (host_sel_i == 4'd2);
        wgt_addr_o   = host_addr_i;
        wgt_wdata_o  = host_wdata_i;

        param_we_o    = host_we_i && ((host_sel_i == 4'd3) || (host_sel_i == 4'd4) || (host_sel_i == 4'd5));
        param_re_o    = host_re_i && ((host_sel_i == 4'd3) || (host_sel_i == 4'd4) || (host_sel_i == 4'd5));
        param_sel_o   = (host_sel_i == 4'd3) ? 2'd0 : (host_sel_i == 4'd4) ? 2'd1 : 2'd2;
        param_addr_o  = host_addr_i;
        param_wdata_o = host_wdata_i;

        ofm_we_o     = host_we_i && ((host_sel_i == 4'd6) || (host_sel_i == 4'd7));
        ofm_re_o     = host_re_i && ((host_sel_i == 4'd6) || (host_sel_i == 4'd7));
        ofm_bank_o   = (host_sel_i == 4'd7);
        ofm_addr_o   = host_addr_i;
        ofm_wdata_o  = host_wdata_i;
    end
endmodule