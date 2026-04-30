`timescale 1ns/1ps

module residual_add #(
    parameter int DATA_W = 8
)(
    input  logic                       en_i,
    input  logic signed [DATA_W-1:0]   a_i,
    input  logic signed [DATA_W-1:0]   b_i,
    output logic signed [DATA_W-1:0]   y_o
);
    logic signed [DATA_W:0] sum_s;
    logic signed [DATA_W:0] max_s;
    logic signed [DATA_W:0] min_s;

    always_comb begin
        sum_s = $signed(a_i) + (en_i ? $signed(b_i) : '0);
        max_s = {2'b00, {(DATA_W-1){1'b1}}};
        min_s = {2'b11, {(DATA_W-1){1'b0}}};

        if (sum_s > max_s) begin
            y_o = {1'b0, {(DATA_W-1){1'b1}}};
        end else if (sum_s < min_s) begin
            y_o = {1'b1, {(DATA_W-1){1'b0}}};
        end else begin
            y_o = sum_s[DATA_W-1:0];
        end
    end
endmodule