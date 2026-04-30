`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 04/30/2026 01:48:13 PM
// Design Name: 
// Module Name: tdp_bram_1clk
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////

module tdp_bram_1clk #(
    parameter int DATA_W = 8,
    parameter int ADDR_W = 12,
    parameter int DEPTH  = 4096
)(
    input  logic clk,

    input  logic                  a_en_i,
    input  logic                  a_we_i,
    input  logic [ADDR_W-1:0]     a_addr_i,
    input  logic [DATA_W-1:0]     a_wdata_i,
    output logic [DATA_W-1:0]     a_rdata_o,

    input  logic                  b_en_i,
    input  logic                  b_we_i,
    input  logic [ADDR_W-1:0]     b_addr_i,
    input  logic [DATA_W-1:0]     b_wdata_i,
    output logic [DATA_W-1:0]     b_rdata_o
);
    (* ram_style = "block" *) logic [DATA_W-1:0] mem [0:DEPTH-1];

    always_ff @(posedge clk) begin
        if (a_en_i) begin
            if (a_we_i) begin
                mem[a_addr_i] <= a_wdata_i;
            end
            a_rdata_o <= mem[a_addr_i];
        end
    end

    always_ff @(posedge clk) begin
        if (b_en_i) begin
            if (b_we_i) begin
                mem[b_addr_i] <= b_wdata_i;
            end
            b_rdata_o <= mem[b_addr_i];
        end
    end
endmodule
