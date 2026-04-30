`timescale 1ns/1ps

module adder_tree #(
    parameter int N     = 8,
    parameter int ACC_W = 32
)(
    input  logic signed [N*ACC_W-1:0] in_flat_i,
    output logic signed [ACC_W-1:0]   sum_o
);
    integer i;
    logic signed [ACC_W-1:0] tmp;

    always_comb begin
        tmp = '0;
        for (i = 0; i < N; i = i + 1) begin
            tmp = tmp + in_flat_i[i*ACC_W +: ACC_W];
        end
        sum_o = tmp;
    end
endmodule