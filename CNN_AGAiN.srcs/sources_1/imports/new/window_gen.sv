`timescale 1ns/1ps

module window_gen_1d #(
    parameter int DATA_W = 8,
    parameter int K_MAX  = 31
)(
    input  logic signed [DATA_W-1:0] sample_i,
    input  logic                     valid_i,
    output logic signed [K_MAX*DATA_W-1:0] window_flat_o
);
    integer k;

    always_comb begin
        window_flat_o = '0;
        for (k = 0; k < K_MAX; k = k + 1) begin
            window_flat_o[k*DATA_W +: DATA_W] = valid_i ? sample_i : '0;
        end
    end
endmodule