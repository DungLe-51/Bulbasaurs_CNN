`timescale 1ns/1ps

module padding_dilation_unit #(
    parameter int LEN_W = 16,
    parameter int K_W   = 8
)(
    input  logic [LEN_W-1:0] out_pos_i,
    input  logic [K_W-1:0]   k_i,
    input  logic [LEN_W-1:0] len_in_i,
    input  logic [K_W-1:0]   stride_i,
    input  logic [K_W-1:0]   dilation_i,
    input  logic [K_W-1:0]   pad_left_i,
    output logic             valid_o,
    output logic [LEN_W-1:0] in_pos_o
);
    logic signed [31:0] raw_s;

    always_comb begin
        raw_s = $signed({1'b0, out_pos_i}) * $signed({1'b0, stride_i})
              + $signed({1'b0, k_i})       * $signed({1'b0, dilation_i})
              - $signed({1'b0, pad_left_i});

        if ((raw_s >= 0) && (raw_s < $signed({1'b0, len_in_i}))) begin
            valid_o  = 1'b1;
            in_pos_o = raw_s[LEN_W-1:0];
        end else begin
            valid_o  = 1'b0;
            in_pos_o = '0;
        end
    end
endmodule