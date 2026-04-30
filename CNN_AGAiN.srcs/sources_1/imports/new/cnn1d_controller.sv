`timescale 1ns/1ps

module cnn1d_controller #(
    parameter int LEN_W = 16,
    parameter int CH_W  = 16,
    parameter int K_W   = 8
)(
    input  logic clk,
    input  logic rst_n,
    input  logic start_i,

    input  logic [1:0]       op_i,
    input  logic [CH_W-1:0]  cin_i,
    input  logic [CH_W-1:0]  cout_i,
    input  logic [LEN_W-1:0] len_in_i,
    input  logic [LEN_W-1:0] len_out_i,
    input  logic [K_W-1:0]   kernel_i,
    input  logic [K_W-1:0]   stride_i,
    input  logic [K_W-1:0]   dilation_i,

    output logic busy_o,
    output logic done_o,
    output logic error_o,

    output logic [CH_W-1:0]  oc_o,
    output logic [CH_W-1:0]  ic_o,
    output logic [LEN_W-1:0] t_o,
    output logic [K_W-1:0]   k_o,

    output logic             bias_read_o,
    output logic             init_acc_o,
    output logic             mac_read_o,
    output logic             mac_accum_o,
    output logic             post_o,
    output logic             ofm_write_o,

    output logic [3:0]       state_dbg_o
);
    localparam logic [1:0] OP_CONV1D    = 2'd0;
    localparam logic [1:0] OP_DWCONV1D  = 2'd1;
    localparam logic [1:0] OP_MAXPOOL1D = 2'd2;

    typedef enum logic [3:0] {
        ST_IDLE      = 4'd0,
        ST_VALIDATE  = 4'd1,
        ST_BIAS_READ = 4'd2,
        ST_BIAS_WAIT = 4'd3,
        ST_INIT_ACC  = 4'd4,
        ST_MAC_READ  = 4'd5,
        ST_MAC_ACCUM = 4'd6,
        ST_POST      = 4'd7,
        ST_OFM_WRITE = 4'd8,
        ST_DONE      = 4'd9,
        ST_ERROR     = 4'd10
    } state_t;

    state_t state_q, state_d;

    logic [CH_W-1:0]  oc_q, ic_q;
    logic [LEN_W-1:0] t_q;
    logic [K_W-1:0]   k_q;

    logic valid_config;
    logic last_inner;
    logic last_output;
    logic conv_like;
    logic maxpool_like;

    assign conv_like    = (op_i == OP_CONV1D) || (op_i == OP_DWCONV1D);
    assign maxpool_like = (op_i == OP_MAXPOOL1D);

    always_comb begin
        valid_config = 1'b1;
        if (!((op_i == OP_CONV1D) || (op_i == OP_DWCONV1D) || (op_i == OP_MAXPOOL1D))) valid_config = 1'b0;
        if (cin_i == '0 || cout_i == '0 || len_in_i == '0 || len_out_i == '0) valid_config = 1'b0;
        if (kernel_i == '0 || stride_i == '0 || dilation_i == '0) valid_config = 1'b0;
        if ((op_i == OP_DWCONV1D) && (cin_i != cout_i)) valid_config = 1'b0;
        if ((op_i == OP_MAXPOOL1D) && (cin_i != cout_i)) valid_config = 1'b0;
    end

    always_comb begin
        if (op_i == OP_CONV1D) begin
            last_inner = (ic_q == (cin_i - 1'b1)) && (k_q == (kernel_i - 1'b1));
        end else begin
            last_inner = (k_q == (kernel_i - 1'b1));
        end
        last_output = (oc_q == (cout_i - 1'b1)) && (t_q == (len_out_i - 1'b1));
    end

    always_comb begin
        state_d = state_q;
        unique case (state_q)
            ST_IDLE: begin
                if (start_i) state_d = ST_VALIDATE;
            end
            ST_VALIDATE: begin
                state_d = valid_config ? (maxpool_like ? ST_INIT_ACC : ST_BIAS_READ) : ST_ERROR;
            end
            ST_BIAS_READ: begin
                state_d = ST_BIAS_WAIT;
            end
            ST_BIAS_WAIT: begin
                state_d = ST_INIT_ACC;
            end
            ST_INIT_ACC: begin
                state_d = ST_MAC_READ;
            end
            ST_MAC_READ: begin
                state_d = ST_MAC_ACCUM;
            end
            ST_MAC_ACCUM: begin
                state_d = last_inner ? ST_POST : ST_MAC_READ;
            end
            ST_POST: begin
                state_d = ST_OFM_WRITE;
            end
            ST_OFM_WRITE: begin
                if (last_output) begin
                    state_d = ST_DONE;
                end else if (t_q == (len_out_i - 1'b1)) begin
                    state_d = maxpool_like ? ST_INIT_ACC : ST_BIAS_READ;
                end else begin
                    state_d = ST_INIT_ACC;
                end
            end
            ST_DONE: begin
                if (!start_i) state_d = ST_IDLE;
            end
            ST_ERROR: begin
                if (!start_i) state_d = ST_IDLE;
            end
            default: begin
                state_d = ST_IDLE;
            end
        endcase
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_q <= ST_IDLE;
            oc_q    <= '0;
            ic_q    <= '0;
            t_q     <= '0;
            k_q     <= '0;
        end else begin
            state_q <= state_d;

            if (state_q == ST_IDLE && start_i) begin
                oc_q <= '0;
                ic_q <= '0;
                t_q  <= '0;
                k_q  <= '0;
            end else if (state_q == ST_INIT_ACC) begin
                k_q <= '0;
                if (op_i == OP_CONV1D) begin
                    ic_q <= '0;
                end else begin
                    ic_q <= oc_q;
                end
            end else if (state_q == ST_MAC_ACCUM) begin
                if (!last_inner) begin
                    if (op_i == OP_CONV1D) begin
                        if (k_q == (kernel_i - 1'b1)) begin
                            k_q  <= '0;
                            ic_q <= ic_q + 1'b1;
                        end else begin
                            k_q <= k_q + 1'b1;
                        end
                    end else begin
                        k_q  <= k_q + 1'b1;
                        ic_q <= oc_q;
                    end
                end
            end else if (state_q == ST_OFM_WRITE) begin
                k_q  <= '0;
                ic_q <= '0;
                if (!last_output) begin
                    if (t_q == (len_out_i - 1'b1)) begin
                        t_q  <= '0;
                        oc_q <= oc_q + 1'b1;
                    end else begin
                        t_q <= t_q + 1'b1;
                    end
                end
            end
        end
    end

    assign busy_o      = (state_q != ST_IDLE) && (state_q != ST_DONE) && (state_q != ST_ERROR);
    assign done_o      = (state_q == ST_DONE);
    assign error_o     = (state_q == ST_ERROR);

    assign oc_o        = oc_q;
    assign ic_o        = (op_i == OP_CONV1D) ? ic_q : oc_q;
    assign t_o         = t_q;
    assign k_o         = k_q;

    assign bias_read_o = (state_q == ST_BIAS_READ);
    assign init_acc_o  = (state_q == ST_INIT_ACC);
    assign mac_read_o  = (state_q == ST_MAC_READ);
    assign mac_accum_o = (state_q == ST_MAC_ACCUM);
    assign post_o      = (state_q == ST_POST);
    assign ofm_write_o = (state_q == ST_OFM_WRITE);

    assign state_dbg_o = state_q;
endmodule