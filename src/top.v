module CNN_top#(
    parameter IMG_SIZE = 30, IMG_FLAT = IMG_SIZE*IMG_SIZE, // chắc là 30x30 :)))
    parameter DATA_WIDTH = 16,// đây là độ lệch
    parameter NUM_F1 = 8, NUM_F2 = 16,// mấy cái này là các thành phần cần phải có
    parameter CONV1_OUT = IMG_SIZE, POOL1_OUT = IMG_SIZE/2,
    parameter CONV2_OUT = POOL1_OUT, POOL2_OUT = CONV2_OUT/2,
    parameter FLAT = POOL2_OUT*POOL2_OUT*NUM_F2, FC_HIDDEN = 128
    /*
    tạm thời kiến trúc sẽ gồm 
    đầu vào ảnh 30x30 (1 ảnh là 1 phổ thu được từ máy OSA)
    rồi vào Conv1 để rút gọn ảnh (thêm các filter nếu cần giống như hồi trước mình thêm filter ấy)
    chia nhỏ thành các kernel 3x3 để quét điểm sáng và điểm tối trên ảnh, sau đó sẽ có 8 filter để tạo ra 8 ảnh mới (có thể là 8 phổ thu được từ máy OSA sau khi đã qua xử lý)
    sau đó sẽ vào Pool1 để giảm kích thước ảnh xuống còn 15x15
    */
)(
    input clk,reset,enable,
    input signed [DATA_WIDTH-1:0] data_in [0:IMG_SIZE*IMG_SIZE-1],
    output signed [DATA_WIDTH-1:0] rho_out,
    output done
);

endmodule   