`timescale 1ns/1ps

module weight_rf #(
    parameter int DATA_W = 8,
    parameter int DEPTH  = 64,
    parameter int ADDR_W = 6
)(
    input  logic clk,
    input  logic rst_n,
    input  logic we_i,
    input  logic [ADDR_W-1:0] waddr_i,
    input  logic signed [DATA_W-1:0] wdata_i,
    input  logic [ADDR_W-1:0] raddr_i,
    output logic signed [DATA_W-1:0] rdata_o
);
    logic signed [DATA_W-1:0] rf [0:DEPTH-1];
    integer i;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < DEPTH; i = i + 1) begin
                rf[i] <= '0;
            end
            rdata_o <= '0;
        end else begin
            if (we_i) begin
                rf[waddr_i] <= wdata_i;
            end
            rdata_o <= rf[raddr_i];
        end
    end
endmodule