# GENETARED BY NNDCT, DO NOT EDIT!

import torch
import pytorch_nndct as py_nndct
class Model_d(torch.nn.Module):
    def __init__(self):
        super(Model_d, self).__init__()
        self.module_0 = py_nndct.nn.Input() #Model_d::input_0
        self.module_1 = py_nndct.nn.ReLU(inplace=False) #Model_d::Model_d/input.1
        self.module_2 = py_nndct.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model_d::Model_d/Conv2d[n_Conv_261]/t_1275
        self.module_3 = py_nndct.nn.ReLU(inplace=False) #Model_d::Model_d/input
        self.module_4 = py_nndct.nn.Conv2d(in_channels=1024, out_channels=29, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model_d::Model_d/Conv2d[n_Conv_263]/1720
        self.module_5 = py_nndct.nn.Module('squeeze') #Model_d::Model_d/1722
        self.module_6 = py_nndct.nn.Module('permute') #Model_d::Model_d/1727

    def forward(self, *args):
        output_module_0 = self.module_0(input=args[0])
        output_module_0 = self.module_1(output_module_0)
        output_module_0 = self.module_2(output_module_0)
        output_module_0 = self.module_3(output_module_0)
        output_module_0 = self.module_4(output_module_0)
        output_module_0 = self.module_5(input=output_module_0, dim=(3))
        output_module_0 = self.module_6(dims=[0,2,1], input=output_module_0)
        return output_module_0
