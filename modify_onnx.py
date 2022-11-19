import onnx
import onnx_graphsurgeon as gs
import numpy as np
import onnxsim

src_onnx_path = "model/model.onnx"
dst_onnx_path="model/model_reshape.onnx"
print("Load onnx model from {}".format(src_onnx_path))
graph = gs.import_onnx(onnx.load(src_onnx_path))
print("Nodes:{}".format(len(graph.nodes)))
graph.fold_constants().cleanup()
names=["read_file_0.tmp_0","read_file_0.tmp_1","read_file_0.tmp_2","read_file_0.tmp_6","read_file_0.tmp_7","read_file_0.tmp_8","read_file_0.tmp_9","read_file_0.tmp_10","read_file_0.tmp_11","read_file_0.tmp_12","read_file_0.tmp_13"]

print(len(graph.inputs))
for i,value in enumerate(graph.inputs):
  if value.name in names:
    graph.inputs[i].shape=value.shape[:2]
    # graph.inputs[i].output = "p2o.Gather.{gather}"
    
gather=0
input=0
for i,value in enumerate(graph.nodes):
  if value.name == f"p2o.Gather.{gather}":
    if gather==3:
      input+=1
    graph.nodes[i].inputs[1]=graph.inputs[input]
    gather+=1
    input+=1
  if gather == 12:
    break

Squeeze=0
for i,value in enumerate(graph.nodes):
  if value.name == f"p2o.Squeeze.{Squeeze}":
    Squeeze+=1
    graph.nodes[i].inputs=[]
    graph.nodes[i].outputs=[]
    if Squeeze==12:
      break
    
graph.cleanup().toposort()
print("Nodes:{}".format(len(graph.nodes)))

print('\nStarting to simplify ONNX...')
onnx_model, check = onnxsim.simplify(gs.export_onnx(graph))
onnx.save(onnx_model, dst_onnx_path)
print("Save modified onnx model to {}".format(dst_onnx_path))