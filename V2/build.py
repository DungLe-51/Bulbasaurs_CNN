import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.change_3d_tensors_to_4d import Change3DTo4DTensors

input_model = "deepdas_model.qonnx"
model_4d = "deepdas_model_4d.qonnx"

print("Đang ép kiểu mạng 1D sang định dạng 2D ảo cho FINN...")
model = ModelWrapper(input_model)
model = model.transform(Change3DTo4DTensors())
model.save(model_4d)
print("Hoàn tất ép kiểu! Đang khởi động lò rèn đúc chip...")

cfg = build_cfg.DataflowBuildConfig(
    output_dir="output_hw",
    synth_clk_period_ns=10.0, 
    fpga_part="xc7vx485tffg1157-2", 
    generate_outputs=[
        build_cfg.DataflowOutputType.STITCHED_IP,
    ]
)

build.build_dataflow_cfg(model_4d, cfg)
