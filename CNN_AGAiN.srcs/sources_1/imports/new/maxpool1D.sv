`timescale 1ns/1ps

module maxpool1d #(
    parameter int DATA_W = 8
)(
    input  logic signed [DATA_W-1:0] cur_max_i,
    input  logic signed [DATA_W-1:0] sample_i,
    input  logic                     valid_i,
    output logic signed [DATA_W-1:0] next_max_o
);
    always_comb begin
        if (!valid_i) begin
            next_max_o = cur_max_i;
        end else if (sample_i > cur_max_i) begin
            next_max_o = sample_i;
        end else begin
            next_max_o = cur_max_i;
        end
    end
endmodule