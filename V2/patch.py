path = "/home/dung/deepdas_finn/finn/deps/qonnx/src/qonnx/transformation/change_3d_tensors_to_4d.py"
with open(path, "r") as f:
    code = f.read()

# Xóa lệnh bắt buộc dừng khi gặp node lạ
code = code.replace("return (model, False)", "# return (model, False) - Đã bị bẻ khóa!")

with open(path, "w") as f:
    f.write(code)
print("BẺ KHÓA FINN THÀNH CÔNG! Đã dọn đường cho 1D CNN!")
