`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 04/29/2026 09:57:30 PM
// Design Name: 
// Module Name: accumulator
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


`timescale 1ns/1ps

module accumulator_file #(
    parameter int DEPTH  = 16,
    parameter int ADDR_W = 4,
    parameter int ACC_W  = 32
)(
    input  logic clk,
    input  logic rst_n,

    input  logic                    clear_i,
    input  logic                    we_i,
    input  logic [ADDR_W-1:0]       waddr_i,
    input  logic signed [ACC_W-1:0] wdata_i,

    input  logic [ADDR_W-1:0]       raddr_i,
    output logic signed [ACC_W-1:0] rdata_o
);
    logic signed [ACC_W-1:0] mem [0:DEPTH-1];
    integer i;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < DEPTH; i = i + 1) begin
                mem[i] <= '0;
            end
            rdata_o <= '0;
        end else begin
            if (clear_i) begin
                for (i = 0; i < DEPTH; i = i + 1) begin
                    mem[i] <= '0;
                end
            end else if (we_i) begin
                mem[waddr_i] <= wdata_i;
            end
            rdata_o <= mem[raddr_i];
        end
    end
endmodule
