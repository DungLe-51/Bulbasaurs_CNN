`timescale 1ns/1ps

module pe_array #(
    parameter int LANES  = 1,
    parameter int DATA_W = 8,
    parameter int WGT_W  = 8,
    parameter int ACC_W  = 32
)(
    input  logic signed [LANES*DATA_W-1:0] act_vec_i,
    input  logic signed [LANES*WGT_W-1:0]  wgt_vec_i,
    output logic signed [ACC_W-1:0]         dot_o
);
    logic signed [LANES*ACC_W-1:0] prod_flat;

    genvar g;
    generate
        for (g = 0; g < LANES; g = g + 1) begin : gen_pe_lanes
            pe_lane #(
                .DATA_W(DATA_W),
                .WGT_W (WGT_W),
                .ACC_W (ACC_W)
            ) u_pe_lane (
                .act_i (act_vec_i[g*DATA_W +: DATA_W]),
                .wgt_i (wgt_vec_i [g*WGT_W  +: WGT_W ]),
                .prod_o(prod_flat[g*ACC_W +: ACC_W])
            );
        end
    endgenerate

    adder_tree #(
        .N    (LANES),
        .ACC_W(ACC_W)
    ) u_adder_tree (
        .in_flat_i(prod_flat),
        .sum_o    (dot_o)
    );
endmodule