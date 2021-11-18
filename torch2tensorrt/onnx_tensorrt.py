import tensorrt as trt
import sys
import argparse
import os
"""
takes in onnx model
converts to tensorrt
tensorrt model input size must be src pth input size 
"""

# def cli():
#     desc = 'Compile Onnx model to TensorRT'
#     parser = argparse.ArgumentParser(description=desc)
#     parser.add_argument('-m', '--model', default='', help='onnx file location')
#     parser.add_argument('-fp', '--floatingpoint', type=int, default=16, help='floating point precision. 16 or 32')
#     parser.add_argument('-o', '--output', default='', help='name of trt output file')
#     args = parser.parse_args()
#     model = args.model or 'yolov5s-simple.onnx'
#     fp = args.floatingpoint
#     if fp != 16 and fp != 32:
#         print('floating point precision must be 16 or 32')
#         sys.exit()
#     output = args.output or 'yolov5s-simple-{}.trt'.format(fp)
#     return {
#         'model': model,
#         'fp': fp,
#         'output': output
#     }

# if __name__ == '__main__':
#     args = cli()
#     batch_size = 4
#     model = '{}'.format(args['model'])
#     output = '{}'.format(args['output'])
#     logger = trt.Logger(trt.Logger.WARNING)
#     explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)  # trt7
#     with trt.Builder(logger) as builder, builder.create_network(explicit_batch) as network, trt.OnnxParser(network, logger) as parser:
#         builder.max_workspace_size = 1 << 28
#         builder.max_batch_size = batch_size
#         if args['fp'] == 16:
#             builder.fp16_mode = True
#         with open(model, 'rb') as f:
#             print('Beginning ONNX file parsing')
#             if not parser.parse(f.read()):
#                 for error in range(parser.num_errors):
#                     print("ERROR", parser.get_error(error))
#         print("num layers:", network.num_layers)
#         network.get_input(0).shape = [batch_size, 3, 608, 608]  # trt7
#         # last_layer = network.get_layer(network.num_layers - 1)
#         # network.mark_output(last_layer.get_output(0))
#         # reshape input from 32 to 1
#         engine = builder.build_cuda_engine(network)
#         with open(output, 'wb') as f:
#             f.write(engine.serialize())
#         print("Completed creating Engine")

def ONNX_to_TensorRT(max_batch_size=1,fp16_mode=True,onnx_model_path=None,trt_engine_path=None):
    """
    生成cudaEngine，并保存引擎文件(仅支持固定输入尺度)
    
    max_batch_size: 默认为1，暂不支持动态batch
    fp16_mode: True则fp16预测
    onnx_model_path: 将加载的onnx权重路径
    trt_engine_path: trt引擎文件保存路径
    """
    # 以trt的Logger为参数，使用builder创建计算图类型INetworkDefinition
    TRT_LOGGER = trt.Logger()

    # 由onnx创建cudaEngine
    # 使用logger创建一个builder
    # builder创建一个计算图 INetworkDefinition
    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    # In TensorRT 7.0, the ONNX parser only supports full-dimensions mode, meaning that your network definition must be created with the explicitBatch flag set. For more information, see Working With Dynamic Shapes.

    with trt.Builder(TRT_LOGGER) as builder, \
            builder.create_network(explicit_batch) as network, \
            trt.OnnxParser(network, TRT_LOGGER) as parser:  # 使用onnx的解析器绑定计算图，后续将通过解析填充计算图
        builder.max_workspace_size = 1 << 30  # 预先分配的工作空间大小,即ICudaEngine执行时GPU最大需要的空间
        builder.max_batch_size = max_batch_size  # 执行时最大可以使用的batchsize
        builder.fp16_mode = fp16_mode

        # 解析onnx文件，填充计算图
        if not os.path.exists(onnx_model_path):
            quit("ONNX file {} not found!".format(onnx_model_path))
        print('loading onnx file from path {} ...'.format(onnx_model_path))
        with open(onnx_model_path, 'rb') as model:  # 二值化的网络结果和参数
            print("Begining onnx file parsing")
            parser.parse(model.read())  # 解析onnx文件
        # parser.parse_from_file(onnx_model_path) # parser还有一个从文件解析onnx的方法

        print("Completed parsing of onnx file")
        # 填充计算图完成后，则使用builder从计算图中创建CudaEngine
        print("Building an engine from file{}' this may take a while...".format(onnx_model_path))

        #################
        output_shape=network.get_layer(network.num_layers - 1).get_output(0).shape
        # network.mark_output(network.get_layer(network.num_layers -1).get_output(0))
        print('output shape:',output_shape)
        engine = builder.build_cuda_engine(network)  # 注意，这里的network是INetworkDefinition类型，即填充后的计算图
        print("Completed creating Engine")

        # 保存engine供以后直接反序列化使用
        with open(trt_engine_path, 'wb') as f:
            f.write(engine.serialize())  # 序列化

        print('TensorRT file in ' + trt_engine_path)
        print('============ONNX->TensorRT SUCCESS============')

if __name__ == '__main__':
    ONNX_to_TensorRT(onnx_model_path='../weights/yolov5n-face.onnx', trt_engine_path='../weights/yolov5n-face.trt')