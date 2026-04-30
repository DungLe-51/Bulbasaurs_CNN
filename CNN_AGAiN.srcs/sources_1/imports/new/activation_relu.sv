`timescale 1ns/1ps

module activation_relu #(
    parameter int DATA_W = 8
)(
    input  logic                       relu_en_i,
    input  logic signed [DATA_W-1:0]   data_i,
    output logic signed [DATA_W-1:0]   data_o
);
    always_comb begin
        if (relu_en_i && data_i[DATA_W-1]) begin
            data_o = '0;
        end else begin
            data_o = data_i;
        end
    end
endmodule