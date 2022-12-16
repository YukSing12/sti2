from typing import Iterable, Tuple

import torch
import numpy as np
import ppq.lib as PFL
from ppq import BaseQuantizer, Operation, OperationQuantizationConfig, TargetPlatform, TorchExecutor, graphwise_error_analyse, BaseGraph, QuantableOperation
from ppq.api import ENABLE_CUDA_KERNEL, load_onnx_graph, register_network_quantizer
from ppq.core import (PASSIVE_OPERATIONS, QuantizationPolicy, QuantizationProperty,
                      QuantizationStates, RoundingPolicy, convert_any_to_numpy)
from ppq.IR import BaseGraph, Operation, QuantableOperation, SearchableGraph
from ppq.lib.quant import Exporter
from ppq.quantization.optim import (LayerwiseEqualizationPass,
                                    LearnedStepSizePass, ParameterQuantizePass,
                                    PassiveParameterQuantizePass,
                                    QuantAlignmentPass, QuantizeFusionPass,
                                    QuantizeSimplifyPass,
                                    RuntimeCalibrationPass)

BATCHSIZE = 1
SEQ_LEN = 128
INPUTS = {'read_file_0.tmp_0': ([BATCHSIZE, SEQ_LEN, 1], np.int64),
          'read_file_0.tmp_1': ([BATCHSIZE, SEQ_LEN, 1], np.int64),
          'read_file_0.tmp_2': ([BATCHSIZE, SEQ_LEN, 1], np.int64),
          'read_file_0.tmp_3': ([BATCHSIZE, SEQ_LEN, 1], np.float32),
          'read_file_0.tmp_6': ([BATCHSIZE, 1, 1], np.int64),
          'read_file_0.tmp_7': ([BATCHSIZE, 1, 1], np.int64),
          'read_file_0.tmp_8': ([BATCHSIZE, 1, 1], np.int64),
          'read_file_0.tmp_9': ([BATCHSIZE, 1, 1], np.int64),
          'read_file_0.tmp_10': ([BATCHSIZE, 1, 1], np.int64),
          'read_file_0.tmp_11': ([BATCHSIZE, 1, 1], np.int64),
          'read_file_0.tmp_12': ([BATCHSIZE, 1, 1], np.int64),
          'read_file_0.tmp_13': ([BATCHSIZE, 1, 1], np.int64)}

DEVICE = 'cuda'
QUANT_PLATFORM = TargetPlatform.TRT_INT8
ONNX_PATH = 'model/model_dymshape_13.onnx'
ONNX_OUTPUT_PATH = 'model/qt_model_dymshape.onnx'
CAL_DATASET = 'data/label.test.txt'

from ppq.quantization.optim import QuantizationOptimizationPass
class FuseQKVOptimPass(QuantizationOptimizationPass):
    def __init__(self, name: str = 'FuseQKV Optimization Pass') -> None:
        super().__init__(name)
    def optimize(self, graph: BaseGraph, **kwargs) -> None:
        search_engine = SearchableGraph(graph=graph)
        matches = search_engine.pattern_matching(
            patterns=[lambda x: True, 'Flatten', 'MatMul', 'Flatten', 'MatMul', 'Flatten', 'MatMul'],
            edges=[[0, 1], [1, 2], [0, 3], [3, 4], [0, 5], [5, 6]], exclusive=False)
        
        visited = set()
        for any, _, mat1, _, mat2, _, mat3 in matches:
            if any in visited: continue
            visited.add(any)
            
            assert isinstance(mat1, QuantableOperation)
            assert isinstance(mat2, QuantableOperation)
            assert isinstance(mat3, QuantableOperation)
            
            input_TQC1 = mat1.config.input_quantization_config[0]
            input_TQC2 = mat2.config.input_quantization_config[0]
            input_TQC3 = mat3.config.input_quantization_config[0]
            
            # 你可以在这里将 input_TQC1.scale 设置成三者中最大的，不过这个操作一般对精度也没啥特别重要的影响
            # 下面的语句将建立 input_TQC1, input_TQC2, input_TQC3 三者的相互关系
            # input_TQC2, input_TQC3 将与 input_TQC1 共用 scale
            input_TQC2.master_by = input_TQC1
            input_TQC3.master_by = input_TQC1

            print(f'{mat1.name}, {mat2.name}, {mat3.name} has been fused.')

# ------------------------------------------------------------
# 在这个例子中我们将向你展示如何量化一个 onnx 模型，执行误差分析，并与 onnxruntime 对齐结果
# 在这个例子中，我们特别地为你展示如何量化一个多输入的模型
# 此时你的 Calibration Dataset 应该是一个 list of dictionary
# ------------------------------------------------------------


def generate_calibration_dataset(graph: BaseGraph, num_of_batches: int = 32) -> Tuple[Iterable[dict], torch.Tensor]:
    dataset = []
    with open(CAL_DATASET, 'r') as fid:
        lines = fid.readlines()
        for line in lines[:num_of_batches]:
            meta_data = line.split(";")
            qid = meta_data[0]
            label = meta_data[1]
            i = 2
            sample = {}
            for name in graph.inputs:
                data = meta_data[i]
                tmp = data.split(":")
                shape = tuple(int(s) for s in tmp[0].split(" "))
                value = [INPUTS[name][1](v) for v in tmp[1].split(" ")]
                value = torch.from_numpy(np.array(value).reshape(shape))
                sample.update({name: value})
                i += 1
            dataset.append(sample)
    return dataset


def collate_fn(batch: dict) -> torch.Tensor:
    return {k: v.to(DEVICE) for k, v in batch.items()}


graph = load_onnx_graph(onnx_import_file=ONNX_PATH)

class FasterTransformerQuantizer(BaseQuantizer):
    # ------------------------------------------------------------
    # quant_operation_types 是一个类型枚举，在这里你需要写下所有该量化器所需要量化的算子
    # ------------------------------------------------------------
    @ property
    def quant_operation_types(self) -> set:
        return {'MatMul'}

    # ------------------------------------------------------------
    # 一旦你确定了那些算子需要量化，则需要在 init_quantize_config 为他们初始化量化信息
    # 然而你需要注意的是，由于手动调度的存在，用户可以强制调度一个类型不在 quant_operation_types 中的算子来到量化平台
    # 我建议你针对这类情况进行回应。或者，在探测到算子类型并非可量化类型后进行报错
    # ------------------------------------------------------------
    def init_quantize_config(self, operation: Operation) -> OperationQuantizationConfig:
        # ------------------------------------------------------------
        # 为矩阵乘算子初始化量化信息，只量化矩阵乘算子的输出
        # ------------------------------------------------------------
        base_quant_config = self.create_default_quant_config(
            policy=self.quantize_policy, rounding=self.rounding_policy,
            op=operation, num_of_bits=8, exponent_bits=0,
            quant_max=127, quant_min=-128,
            observer_algorithm='percentile'
        )
        if operation.type == 'MatMul':
            if operation.inputs[1].is_parameter:
                gemm_weight_config = base_quant_config.input_quantization_config[1]
                gemm_weight_config.policy = QuantizationPolicy(
                    QuantizationProperty.SYMMETRICAL +
                    QuantizationProperty.LINEAR +
                    QuantizationProperty.PER_CHANNEL
                )
                gemm_weight_config.channel_axis = 0
                gemm_weight_config.observer_algorithm = 'minmax'
            
        if operation.type in PASSIVE_OPERATIONS:
            # Those op are not active op.
            base_quant_config.is_active_quant_op = False
        return base_quant_config

    @ property
    def quantize_policy(self) -> QuantizationPolicy:
        return QuantizationPolicy(
            QuantizationProperty.SYMMETRICAL +
            QuantizationProperty.LINEAR +
            QuantizationProperty.PER_TENSOR
        )

    @ property
    def target_platform(self) -> TargetPlatform:
        return TargetPlatform.EXTENSION

    @ property
    def rounding_policy(self) -> RoundingPolicy:
        return RoundingPolicy.ROUND_HALF_EVEN

    @ property
    def activation_fusion_types(self) -> set:
        return {}
register_network_quantizer(FasterTransformerQuantizer, TargetPlatform.EXTENSION)

# ------------------------------------------------------------
# 我们首先进行标准的量化流程，为所有算子初始化量化信息，并进行 Calibration
# ------------------------------------------------------------

quantizer   = FasterTransformerQuantizer(graph)
dispatching = PFL.Dispatcher(graph=graph).dispatch(          # 生成调度表
    quant_types=quantizer.quant_operation_types)

# 为算子初始化量化信息
for op in graph.operations.values():
    quantizer.quantize_operation(
        op_name = op.name, platform = dispatching[op.name])

# 初始化执行器
executor = TorchExecutor(graph=graph, device=DEVICE)
calibration_dataset = generate_calibration_dataset(graph)
executor.tracing_operation_meta(inputs=calibration_dataset[0])
executor.load_graph(graph=graph)

# ------------------------------------------------------------
# 创建优化管线，由于后续还要继续训练我们的模型，我们不能在此处调用
# ParameterBakingPass()，一旦模型权重完成烘焙，则它们不能被进一步调整
# ------------------------------------------------------------
pipeline = PFL.Pipeline([
    QuantizeSimplifyPass(),
    QuantizeFusionPass(
        activation_type=quantizer.activation_fusion_types),
    ParameterQuantizePass(),
    RuntimeCalibrationPass(),
    PassiveParameterQuantizePass(),
    QuantAlignmentPass(force_overlap=True),
    FuseQKVOptimPass()
])

with ENABLE_CUDA_KERNEL():
    # 调用管线完成量化
    pipeline.optimize(
        graph=graph, dataloader=calibration_dataset, verbose=True, 
        calib_steps=32, collate_fn=collate_fn, executor=executor)

    graphwise_error_analyse(
        graph=graph, running_device=DEVICE, dataloader=calibration_dataset,
        collate_fn=collate_fn, steps=8)

    exporter = Exporter(platform=TargetPlatform.ONNXRUNTIME)
    exporter.export(file_path=ONNX_OUTPUT_PATH, graph=graph)
