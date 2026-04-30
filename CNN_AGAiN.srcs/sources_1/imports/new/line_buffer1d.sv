`timescale 1ns/1ps

module line_buffer_1d #(
    parameter int DATA_W = 8,
    parameter int DEPTH  = 64
)(
    input  logic clk,
    input  logic rst_n,
    input  logic clear_i,
    input  logic push_i,
    input  logic signed [DATA_W-1:0] sample_i,
    input  logic [$clog2(DEPTH)-1:0] tap_i,
    output logic signed [DATA_W-1:0] tap_o
);
    logic signed [DATA_W-1:0] shift_mem [0:DEPTH-1];
    integer i;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < DEPTH; i = i + 1) begin
                shift_mem[i] <= '0;
            end
        end else if (clear_i) begin
            for (i = 0; i < DEPTH; i = i + 1) begin
                shift_mem[i] <= '0;
            end
        end else if (push_i) begin
            shift_mem[0] <= sample_i;
            for (i = 1; i < DEPTH; i = i + 1) begin
                shift_mem[i] <= shift_mem[i-1];
            end
        end
    end

    assign tap_o = shift_mem[tap_i];
endmodule