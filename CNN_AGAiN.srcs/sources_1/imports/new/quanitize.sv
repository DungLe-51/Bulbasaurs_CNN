`timescale 1ns/1ps

module requantize #(
    parameter int ACC_W = 32,
    parameter int OUT_W = 8
)(
    input  logic signed [ACC_W-1:0] acc_i,
    input  logic signed [31:0]      mult_i,
    input  logic [4:0]              shift_i,
    output logic signed [OUT_W-1:0] data_o
);
    logic signed [63:0] prod_s;
    logic signed [63:0] round_s;
    logic signed [63:0] shifted_s;
    logic signed [63:0] max_s;
    logic signed [63:0] min_s;

    always_comb begin
        prod_s = $signed(acc_i) * $signed(mult_i);

        if (shift_i == 5'd0) begin
            round_s = prod_s;
        end else if (prod_s >= 0) begin
            round_s = prod_s + (64'sd1 <<< (shift_i - 1));
        end else begin
            round_s = prod_s - (64'sd1 <<< (shift_i - 1));
        end

        shifted_s = round_s >>> shift_i;
        max_s = (64'sd1 <<< (OUT_W-1)) - 64'sd1;
        min_s = -(64'sd1 <<< (OUT_W-1));

        if (shifted_s > max_s) begin
            data_o = {1'b0, {(OUT_W-1){1'b1}}};
        end else if (shifted_s < min_s) begin
            data_o = {1'b1, {(OUT_W-1){1'b0}}};
        end else begin
            data_o = shifted_s[OUT_W-1:0];
        end
    end
endmodule