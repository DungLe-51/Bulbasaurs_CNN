`timescale 1ns/1ps

module writeback_unit #(
    parameter int DATA_W = 8,
    parameter int ADDR_W = 16
)(
    input  logic                     valid_i,
    input  logic [ADDR_W-1:0]        addr_i,
    input  logic signed [DATA_W-1:0] data_i,
    output logic                     mem_we_o,
    output logic [ADDR_W-1:0]        mem_addr_o,
    output logic signed [DATA_W-1:0] mem_wdata_o
);
    assign mem_we_o    = valid_i;
    assign mem_addr_o  = addr_i;
    assign mem_wdata_o = data_i;
endmodule