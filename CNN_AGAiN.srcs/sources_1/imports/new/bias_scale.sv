`timescale 1ns/1ps

module bias_scale_rf #(
    parameter int DEPTH  = 16,
    parameter int ADDR_W = 4
)(
    input  logic clk,
    input  logic rst_n,
    input  logic we_i,
    input  logic [ADDR_W-1:0] waddr_i,
    input  logic signed [31:0] bias_i,
    input  logic signed [31:0] mult_i,
    input  logic [31:0]        shift_i,
    input  logic [ADDR_W-1:0] raddr_i,
    output logic signed [31:0] bias_o,
    output logic signed [31:0] mult_o,
    output logic [31:0]        shift_o
);
    logic signed [31:0] bias_rf  [0:DEPTH-1];
    logic signed [31:0] mult_rf  [0:DEPTH-1];
    logic [31:0]        shift_rf [0:DEPTH-1];
    integer i;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < DEPTH; i = i + 1) begin
                bias_rf [i] <= '0;
                mult_rf [i] <= 32'sd1;
                shift_rf[i] <= '0;
            end
            bias_o  <= '0;
            mult_o  <= 32'sd1;
            shift_o <= '0;
        end else begin
            if (we_i) begin
                bias_rf [waddr_i] <= bias_i;
                mult_rf [waddr_i] <= mult_i;
                shift_rf[waddr_i] <= shift_i;
            end
            bias_o  <= bias_rf [raddr_i];
            mult_o  <= mult_rf [raddr_i];
            shift_o <= shift_rf[raddr_i];
        end
    end
endmodule