`timescale 1ns/1ps

module tile_scheduler #(
    parameter int W = 16
)(
    input  logic [W-1:0] total_len_i,
    input  logic [W-1:0] tile_len_i,
    input  logic [W-1:0] tile_idx_i,
    output logic [W-1:0] tile_start_o,
    output logic [W-1:0] tile_valid_len_o,
    output logic         tile_last_o
);
    logic [2*W-1:0] start_long;
    logic [W-1:0] remaining;

    always_comb begin
        start_long   = tile_idx_i * tile_len_i;
        tile_start_o = start_long[W-1:0];

        if (start_long[W-1:0] >= total_len_i) begin
            remaining = '0;
        end else begin
            remaining = total_len_i - start_long[W-1:0];
        end

        if (remaining > tile_len_i) begin
            tile_valid_len_o = tile_len_i;
        end else begin
            tile_valid_len_o = remaining;
        end

        tile_last_o = (remaining <= tile_len_i);
    end
endmodule