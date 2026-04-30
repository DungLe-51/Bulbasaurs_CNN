`timescale 1ns/1ps

module loop_counter #(
    parameter int W = 16
)(
    input  logic clk,
    input  logic rst_n,
    input  logic clear_i,
    input  logic inc_i,
    input  logic [W-1:0] max_i,
    output logic [W-1:0] count_o,
    output logic         last_o
);
    assign last_o = (count_o == (max_i - {{(W-1){1'b0}}, 1'b1}));

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            count_o <= '0;
        end else if (clear_i) begin
            count_o <= '0;
        end else if (inc_i) begin
            if (last_o) begin
                count_o <= '0;
            end else begin
                count_o <= count_o + {{(W-1){1'b0}}, 1'b1};
            end
        end
    end
endmodule