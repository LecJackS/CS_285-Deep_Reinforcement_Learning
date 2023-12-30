import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True
data = torch.randn([32, 64, 9, 9], dtype=torch.float, device='cuda', requires_grad=True)
net = torch.nn.Conv2d(64, 64, kernel_size=[3, 3], padding=[0, 0], stride=[1, 1], dilation=[1, 1], groups=1)
net = net.cuda().float()
out = net(data)
out.backward(torch.randn_like(out))
torch.cuda.synchronize()
#
# ConvolutionParams
#     data_type = CUDNN_DATA_FLOAT
#     padding = [0, 0, 0]
#     stride = [1, 1, 0]
#     dilation = [1, 1, 0]
#     groups = 1
#     deterministic = false
#     allow_tf32 = true
# input: TensorDescriptor 0x55d19ca42a60
#     type = CUDNN_DATA_FLOAT
#     nbDims = 4
#     dimA = 32, 64, 9, 9,
#     strideA = 5184, 81, 9, 1,
# output: TensorDescriptor 0x55d1dfb03540
#     type = CUDNN_DATA_FLOAT
#     nbDims = 4
#     dimA = 32, 64, 7, 7,
#     strideA = 3136, 49, 7, 1,
# weight: FilterDescriptor 0x55d19ca46100
#     type = CUDNN_DATA_FLOAT
#     tensor_format = CUDNN_TENSOR_NCHW
#     nbDims = 4
#     dimA = 64, 64, 3, 3,
# Pointer addresses:
#     input: 0x701820000
#     output: 0x701796000
#     weight: 0x701199c00
# Forward algorithm: 1

