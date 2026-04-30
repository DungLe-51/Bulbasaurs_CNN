`timescale 1ns/1ps

module bank_arbiter(
    input  logic [3:0]  host_sel_i,
    input  logic [31:0] ifm_rdata_i,
    input  logic [31:0] wgt_rdata_i,
    input  logic [31:0] param_rdata_i,
    input  logic [31:0] ofm_rdata_i,
    output logic [31:0] host_rdata_o
);
    always_comb begin
        unique case (host_sel_i)
            4'd0, 4'd1: host_rdata_o = ifm_rdata_i;
            4'd2:       host_rdata_o = wgt_rdata_i;
            4'd3,
            4'd4,
            4'd5:       host_rdata_o = param_rdata_i;
            4'd6, 4'd7: host_rdata_o = ofm_rdata_i;
            default:    host_rdata_o = 32'h0;
        endcase
    end
endmodule