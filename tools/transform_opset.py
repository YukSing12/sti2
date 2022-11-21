import onnx

model_path = './model/model.onnx'
original_model = onnx.load(model_path)
print(original_model.opset_import[0])
original_model.opset_import[0].version = 13
original_model.ir_version = 6

onnx.save(original_model, model_path)
