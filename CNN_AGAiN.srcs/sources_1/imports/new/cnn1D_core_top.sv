`timescale 1ns/1ps

module cnn1d_core_top #(
    parameter int DATA_W       = 8,
    parameter int WGT_W        = 8,
    parameter int ACC_W        = 32,
    parameter int ADDR_W       = 16,
    parameter int PARAM_ADDR_W = 12,
    parameter int LEN_W        = 16,
    parameter int CH_W         = 16,
    parameter int K_W          = 8,
    parameter int IFM_DEPTH    = 65536,
    parameter int OFM_DEPTH    = 65536,
    parameter int WGT_DEPTH    = 65536,
    parameter int PARAM_DEPTH  = 4096
)(
    input  logic clk,
    input  logic rst_n,

    input  logic start_i,
    output logic busy_o,
    output logic done_o,
    output logic error_o,
    output logic [3:0] state_dbg_o,

    input  logic desc_we_i,
    input  logic [1:0]        desc_op_i,
    input  logic              desc_ifm_bank_i,
    input  logic              desc_ofm_bank_i,
    input  logic              desc_relu_en_i,
    input  logic [CH_W-1:0]   desc_cin_i,
    input  logic [CH_W-1:0]   desc_cout_i,
    input  logic [LEN_W-1:0]  desc_len_in_i,
    input  logic [LEN_W-1:0]  desc_len_out_i,
    input  logic [K_W-1:0]    desc_kernel_i,
    input  logic [K_W-1:0]    desc_stride_i,
    input  logic [K_W-1:0]    desc_dilation_i,
    input  logic [K_W-1:0]    desc_pad_left_i,
    input  logic [ADDR_W-1:0] desc_ifm_base_i,
    input  logic [ADDR_W-1:0] desc_ofm_base_i,
    input  logic [ADDR_W-1:0] desc_wgt_base_i,
    input  logic [ADDR_W-1:0] desc_param_base_i,

    input  logic              host_we_i,
    input  logic              host_re_i,
    input  logic [3:0]        host_sel_i,
    input  logic [ADDR_W-1:0] host_addr_i,
    input  logic [31:0]       host_wdata_i,
    output logic [31:0]       host_rdata_o
);
    localparam logic [1:0] OP_CONV1D    = 2'd0;
    localparam logic [1:0] OP_DWCONV1D  = 2'd1;
    localparam logic [1:0] OP_MAXPOOL1D = 2'd2;

    logic [1:0]        op_q;
    logic              ifm_bank_q;
    logic              ofm_bank_q;
    logic              relu_en_q;
    logic [CH_W-1:0]   cin_q;
    logic [CH_W-1:0]   cout_q;
    logic [LEN_W-1:0]  len_in_q;
    logic [LEN_W-1:0]  len_out_q;
    logic [K_W-1:0]    kernel_q;
    logic [K_W-1:0]    stride_q;
    logic [K_W-1:0]    dilation_q;
    logic [K_W-1:0]    pad_left_q;
    logic [ADDR_W-1:0] ifm_base_q;
    logic [ADDR_W-1:0] ofm_base_q;
    logic [ADDR_W-1:0] wgt_base_q;
    logic [ADDR_W-1:0] param_base_q;

    logic [CH_W-1:0]   oc;
    logic [CH_W-1:0]   ic;
    logic [LEN_W-1:0]  t;
    logic [K_W-1:0]    k;

    logic bias_read;
    logic init_acc;
    logic mac_read;
    logic mac_accum;
    logic post;
    logic ofm_write;

    logic pad_valid;
    logic [LEN_W-1:0] in_pos;
    logic [ADDR_W-1:0] ifm_core_addr;
    logic [ADDR_W-1:0] ofm_core_addr;
    logic [ADDR_W-1:0] wgt_core_addr;

    logic ifm_host_we, ifm_host_re, ifm_host_bank;
    logic [ADDR_W-1:0] ifm_host_addr;
    logic [31:0] ifm_host_wdata;
    logic [31:0] ifm_host_rdata;

    logic wgt_host_we, wgt_host_re;
    logic [ADDR_W-1:0] wgt_host_addr;
    logic [31:0] wgt_host_wdata;
    logic [31:0] wgt_host_rdata;

    logic param_host_we, param_host_re;
    logic [1:0] param_host_sel;
    logic [ADDR_W-1:0] param_host_addr_full;
    logic [31:0] param_host_wdata;
    logic [31:0] param_host_rdata;

    logic ofm_host_we, ofm_host_re, ofm_host_bank;
    logic [ADDR_W-1:0] ofm_host_addr;
    logic [31:0] ofm_host_wdata;
    logic [31:0] ofm_host_rdata;

    logic signed [DATA_W-1:0] ifm_rdata;
    logic signed [WGT_W-1:0]  wgt_rdata;
    logic signed [DATA_W-1:0] ofm_core_rdata_unused;

    logic signed [31:0] bias_rdata;
    logic signed [31:0] mult_rdata;
    logic [31:0]        shift_rdata;
    logic [ADDR_W-1:0]  param_core_addr_full;
    logic [PARAM_ADDR_W-1:0] param_core_addr;

    logic signed [ACC_W-1:0] acc_q;
    logic signed [ACC_W-1:0] dot;
    logic signed [DATA_W-1:0] requant_data;
    logic signed [DATA_W-1:0] relu_data;
    logic signed [DATA_W-1:0] pool_next;
    logic signed [DATA_W-1:0] post_data_q;

    logic wr_we;
    logic [ADDR_W-1:0] wr_addr;
    logic signed [DATA_W-1:0] wr_data;

    assign param_core_addr_full = param_base_q + oc;
    assign param_core_addr = param_core_addr_full[PARAM_ADDR_W-1:0];

    layer_desc_regs #(
        .ADDR_W(ADDR_W), .LEN_W(LEN_W), .CH_W(CH_W), .K_W(K_W)
    ) u_layer_desc_regs (
        .clk(clk), .rst_n(rst_n), .desc_we_i(desc_we_i),
        .op_i(desc_op_i), .ifm_bank_i(desc_ifm_bank_i), .ofm_bank_i(desc_ofm_bank_i), .relu_en_i(desc_relu_en_i),
        .cin_i(desc_cin_i), .cout_i(desc_cout_i), .len_in_i(desc_len_in_i), .len_out_i(desc_len_out_i),
        .kernel_i(desc_kernel_i), .stride_i(desc_stride_i), .dilation_i(desc_dilation_i), .pad_left_i(desc_pad_left_i),
        .ifm_base_i(desc_ifm_base_i), .ofm_base_i(desc_ofm_base_i), .wgt_base_i(desc_wgt_base_i), .param_base_i(desc_param_base_i),
        .op_o(op_q), .ifm_bank_o(ifm_bank_q), .ofm_bank_o(ofm_bank_q), .relu_en_o(relu_en_q),
        .cin_o(cin_q), .cout_o(cout_q), .len_in_o(len_in_q), .len_out_o(len_out_q),
        .kernel_o(kernel_q), .stride_o(stride_q), .dilation_o(dilation_q), .pad_left_o(pad_left_q),
        .ifm_base_o(ifm_base_q), .ofm_base_o(ofm_base_q), .wgt_base_o(wgt_base_q), .param_base_o(param_base_q)
    );

    cnn1d_controller #(
        .LEN_W(LEN_W), .CH_W(CH_W), .K_W(K_W)
    ) u_controller (
        .clk(clk), .rst_n(rst_n), .start_i(start_i),
        .op_i(op_q), .cin_i(cin_q), .cout_i(cout_q), .len_in_i(len_in_q), .len_out_i(len_out_q),
        .kernel_i(kernel_q), .stride_i(stride_q), .dilation_i(dilation_q),
        .busy_o(busy_o), .done_o(done_o), .error_o(error_o),
        .oc_o(oc), .ic_o(ic), .t_o(t), .k_o(k),
        .bias_read_o(bias_read), .init_acc_o(init_acc), .mac_read_o(mac_read), .mac_accum_o(mac_accum),
        .post_o(post), .ofm_write_o(ofm_write), .state_dbg_o(state_dbg_o)
    );

    padding_dilation_unit #(
        .LEN_W(LEN_W), .K_W(K_W)
    ) u_padding_dilation (
        .out_pos_i(t), .k_i(k), .len_in_i(len_in_q), .stride_i(stride_q),
        .dilation_i(dilation_q), .pad_left_i(pad_left_q), .valid_o(pad_valid), .in_pos_o(in_pos)
    );

    ifm_addr_gen_1d #(
        .ADDR_W(ADDR_W), .LEN_W(LEN_W), .CH_W(CH_W)
    ) u_ifm_addr_gen (
        .base_i(ifm_base_q), .channel_i(ic), .pos_i(in_pos), .len_i(len_in_q), .addr_o(ifm_core_addr)
    );

    ofm_addr_gen_1d #(
        .ADDR_W(ADDR_W), .LEN_W(LEN_W), .CH_W(CH_W)
    ) u_ofm_addr_gen (
        .base_i(ofm_base_q), .channel_i(oc), .pos_i(t), .len_i(len_out_q), .addr_o(ofm_core_addr)
    );

    weight_addr_gen #(
        .ADDR_W(ADDR_W), .CH_W(CH_W), .K_W(K_W)
    ) u_weight_addr_gen (
        .op_i(op_q), .base_i(wgt_base_q), .oc_i(oc), .ic_i(ic), .cin_i(cin_q), .k_i(k), .kernel_i(kernel_q), .addr_o(wgt_core_addr)
    );

    bank_loader #(
        .ADDR_W(ADDR_W)
    ) u_bank_loader (
        .host_we_i(host_we_i), .host_re_i(host_re_i), .host_sel_i(host_sel_i), .host_addr_i(host_addr_i), .host_wdata_i(host_wdata_i),
        .ifm_we_o(ifm_host_we), .ifm_re_o(ifm_host_re), .ifm_bank_o(ifm_host_bank), .ifm_addr_o(ifm_host_addr), .ifm_wdata_o(ifm_host_wdata),
        .wgt_we_o(wgt_host_we), .wgt_re_o(wgt_host_re), .wgt_addr_o(wgt_host_addr), .wgt_wdata_o(wgt_host_wdata),
        .param_we_o(param_host_we), .param_re_o(param_host_re), .param_sel_o(param_host_sel), .param_addr_o(param_host_addr_full), .param_wdata_o(param_host_wdata),
        .ofm_we_o(ofm_host_we), .ofm_re_o(ofm_host_re), .ofm_bank_o(ofm_host_bank), .ofm_addr_o(ofm_host_addr), .ofm_wdata_o(ofm_host_wdata)
    );

    ifm_pingpong_bank #(
        .DATA_W(DATA_W), .ADDR_W(ADDR_W), .DEPTH(IFM_DEPTH)
    ) u_ifm_bank (
        .clk(clk), .rst_n(rst_n),
        .host_we_i(ifm_host_we), .host_re_i(ifm_host_re), .host_bank_i(ifm_host_bank), .host_addr_i(ifm_host_addr), .host_wdata_i(ifm_host_wdata), .host_rdata_o(ifm_host_rdata),
        .core_re_i(mac_read && pad_valid), .core_bank_i(ifm_bank_q), .core_addr_i(ifm_core_addr), .core_rdata_o(ifm_rdata)
    );

    weight_sram_bank #(
        .DATA_W(WGT_W), .ADDR_W(ADDR_W), .DEPTH(WGT_DEPTH)
    ) u_weight_bank (
        .clk(clk), .rst_n(rst_n),
        .host_we_i(wgt_host_we), .host_re_i(wgt_host_re), .host_addr_i(wgt_host_addr), .host_wdata_i(wgt_host_wdata), .host_rdata_o(wgt_host_rdata),
        .core_re_i(mac_read && pad_valid && (op_q != OP_MAXPOOL1D)), .core_addr_i(wgt_core_addr), .core_rdata_o(wgt_rdata)
    );

    bias_scale_bank #(
        .ADDR_W(PARAM_ADDR_W), .DEPTH(PARAM_DEPTH)
    ) u_bias_scale_bank (
        .clk(clk), .rst_n(rst_n),
        .host_we_i(param_host_we), .host_re_i(param_host_re), .host_param_sel_i(param_host_sel),
        .host_addr_i(param_host_addr_full[PARAM_ADDR_W-1:0]), .host_wdata_i(param_host_wdata), .host_rdata_o(param_host_rdata),
        .core_re_i(bias_read), .core_addr_i(param_core_addr),
        .bias_o(bias_rdata), .mult_o(mult_rdata), .shift_o(shift_rdata)
    );

    pe_array #(
        .LANES(1), .DATA_W(DATA_W), .WGT_W(WGT_W), .ACC_W(ACC_W)
    ) u_pe_array (
        .act_vec_i(ifm_rdata), .wgt_vec_i(wgt_rdata), .dot_o(dot)
    );

    requantize #(
        .ACC_W(ACC_W), .OUT_W(DATA_W)
    ) u_requantize (
        .acc_i(acc_q), .mult_i(mult_rdata), .shift_i(shift_rdata[4:0]), .data_o(requant_data)
    );

    activation_relu #(
        .DATA_W(DATA_W)
    ) u_activation_relu (
        .relu_en_i(relu_en_q), .data_i(requant_data), .data_o(relu_data)
    );

    maxpool1d #(
        .DATA_W(DATA_W)
    ) u_maxpool1d (
        .cur_max_i(acc_q[DATA_W-1:0]), .sample_i(ifm_rdata), .valid_i(pad_valid), .next_max_o(pool_next)
    );

    writeback_unit #(
        .DATA_W(DATA_W), .ADDR_W(ADDR_W)
    ) u_writeback_unit (
        .valid_i(ofm_write), .addr_i(ofm_core_addr), .data_i(post_data_q),
        .mem_we_o(wr_we), .mem_addr_o(wr_addr), .mem_wdata_o(wr_data)
    );

    ofm_pingpong_bank #(
        .DATA_W(DATA_W), .ADDR_W(ADDR_W), .DEPTH(OFM_DEPTH)
    ) u_ofm_bank (
        .clk(clk), .rst_n(rst_n),
        .host_we_i(ofm_host_we), .host_re_i(ofm_host_re), .host_bank_i(ofm_host_bank), .host_addr_i(ofm_host_addr), .host_wdata_i(ofm_host_wdata), .host_rdata_o(ofm_host_rdata),
        .core_we_i(wr_we), .core_re_i(1'b0), .core_bank_i(ofm_bank_q), .core_addr_i(wr_addr), .core_wdata_i(wr_data), .core_rdata_o(ofm_core_rdata_unused)
    );

    bank_arbiter u_bank_arbiter (
        .host_sel_i(host_sel_i), .ifm_rdata_i(ifm_host_rdata), .wgt_rdata_i(wgt_host_rdata),
        .param_rdata_i(param_host_rdata), .ofm_rdata_i(ofm_host_rdata), .host_rdata_o(host_rdata_o)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc_q       <= '0;
            post_data_q <= '0;
        end else begin
            if (init_acc) begin
                if (op_q == OP_MAXPOOL1D) begin
                    acc_q <= {{(ACC_W-DATA_W){1'b1}}, 1'b1, {(DATA_W-1){1'b0}}};
                end else begin
                    acc_q <= bias_rdata[ACC_W-1:0];
                end
            end else if (mac_accum) begin
                if (op_q == OP_MAXPOOL1D) begin
                    if (pad_valid) begin
                        acc_q <= {{(ACC_W-DATA_W){pool_next[DATA_W-1]}}, pool_next};
                    end
                end else begin
                    if (pad_valid) begin
                        acc_q <= acc_q + dot;
                    end
                end
            end

            if (post) begin
                if (op_q == OP_MAXPOOL1D) begin
                    post_data_q <= acc_q[DATA_W-1:0];
                end else begin
                    post_data_q <= relu_data;
                end
            end
        end
    end
endmodule