# GENETARED BY NNDCT, DO NOT EDIT!

import torch
import pytorch_nndct as py_nndct
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.module_0 = py_nndct.nn.Input() #Model::input_0
        self.module_1 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[33, 1], stride=[2, 1], padding=[16, 0], dilation=[1, 1], groups=64, bias=False) #Model::Model/Conv2d[n_Conv_0]/input.2
        self.module_2 = py_nndct.nn.Conv2d(in_channels=64, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_1]/t_999
        self.module_3 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.3
        self.module_4 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[33, 1], stride=[1, 1], padding=[16, 0], dilation=[1, 1], groups=256, bias=False) #Model::Model/Conv2d[n_Conv_3]/input.4
        self.module_5 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_4]/t_1002
        self.module_6 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.5
        self.module_7 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[33, 1], stride=[1, 1], padding=[16, 0], dilation=[1, 1], groups=256, bias=False) #Model::Model/Conv2d[n_Conv_6]/input.6
        self.module_8 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_7]/t_1005
        self.module_9 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.7
        self.module_10 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[33, 1], stride=[1, 1], padding=[16, 0], dilation=[1, 1], groups=256, bias=False) #Model::Model/Conv2d[n_Conv_9]/input.8
        self.module_11 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_10]/t_1008
        self.module_12 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.9
        self.module_13 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[33, 1], stride=[1, 1], padding=[16, 0], dilation=[1, 1], groups=256, bias=False) #Model::Model/Conv2d[n_Conv_12]/input.10
        self.module_14 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_13]/t_1011
        self.module_15 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.11
        self.module_16 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[33, 1], stride=[1, 1], padding=[16, 0], dilation=[1, 1], groups=256, bias=False) #Model::Model/Conv2d[n_Conv_15]/input.12
        self.module_17 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_16]/8770
        self.module_18 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_17]/8789
        self.module_19 = py_nndct.nn.Add() #Model::Model/t_665
        self.module_20 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.13
        self.module_21 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[33, 1], stride=[1, 1], padding=[16, 0], dilation=[1, 1], groups=256, bias=False) #Model::Model/Conv2d[n_Conv_20]/input.14
        self.module_22 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_21]/t_1020
        self.module_23 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.15
        self.module_24 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[33, 1], stride=[1, 1], padding=[16, 0], dilation=[1, 1], groups=256, bias=False) #Model::Model/Conv2d[n_Conv_23]/input.16
        self.module_25 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_24]/t_1023
        self.module_26 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.17
        self.module_27 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[33, 1], stride=[1, 1], padding=[16, 0], dilation=[1, 1], groups=256, bias=False) #Model::Model/Conv2d[n_Conv_26]/input.18
        self.module_28 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_27]/t_1026
        self.module_29 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.19
        self.module_30 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[33, 1], stride=[1, 1], padding=[16, 0], dilation=[1, 1], groups=256, bias=False) #Model::Model/Conv2d[n_Conv_29]/input.20
        self.module_31 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_30]/t_1029
        self.module_32 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.21
        self.module_33 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[33, 1], stride=[1, 1], padding=[16, 0], dilation=[1, 1], groups=256, bias=False) #Model::Model/Conv2d[n_Conv_32]/input.22
        self.module_34 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_33]/8991
        self.module_35 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_34]/9010
        self.module_36 = py_nndct.nn.Add() #Model::Model/t_688
        self.module_37 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.23
        self.module_38 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[33, 1], stride=[1, 1], padding=[16, 0], dilation=[1, 1], groups=256, bias=False) #Model::Model/Conv2d[n_Conv_37]/input.24
        self.module_39 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_38]/t_1038
        self.module_40 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.25
        self.module_41 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[33, 1], stride=[1, 1], padding=[16, 0], dilation=[1, 1], groups=256, bias=False) #Model::Model/Conv2d[n_Conv_40]/input.26
        self.module_42 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_41]/t_1041
        self.module_43 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.27
        self.module_44 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[33, 1], stride=[1, 1], padding=[16, 0], dilation=[1, 1], groups=256, bias=False) #Model::Model/Conv2d[n_Conv_43]/input.28
        self.module_45 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_44]/t_1044
        self.module_46 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.29
        self.module_47 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[33, 1], stride=[1, 1], padding=[16, 0], dilation=[1, 1], groups=256, bias=False) #Model::Model/Conv2d[n_Conv_46]/input.30
        self.module_48 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_47]/t_1047
        self.module_49 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.31
        self.module_50 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[33, 1], stride=[1, 1], padding=[16, 0], dilation=[1, 1], groups=256, bias=False) #Model::Model/Conv2d[n_Conv_49]/input.32
        self.module_51 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_50]/9212
        self.module_52 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_51]/9231
        self.module_53 = py_nndct.nn.Add() #Model::Model/t_711
        self.module_54 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.33
        self.module_55 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[39, 1], stride=[1, 1], padding=[19, 0], dilation=[1, 1], groups=256, bias=False) #Model::Model/Conv2d[n_Conv_54]/input.34
        self.module_56 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_55]/t_1056
        self.module_57 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.35
        self.module_58 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[39, 1], stride=[1, 1], padding=[19, 0], dilation=[1, 1], groups=256, bias=False) #Model::Model/Conv2d[n_Conv_57]/input.36
        self.module_59 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_58]/t_1059
        self.module_60 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.37
        self.module_61 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[39, 1], stride=[1, 1], padding=[19, 0], dilation=[1, 1], groups=256, bias=False) #Model::Model/Conv2d[n_Conv_60]/input.38
        self.module_62 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_61]/t_1062
        self.module_63 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.39
        self.module_64 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[39, 1], stride=[1, 1], padding=[19, 0], dilation=[1, 1], groups=256, bias=False) #Model::Model/Conv2d[n_Conv_63]/input.40
        self.module_65 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_64]/t_1065
        self.module_66 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.41
        self.module_67 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[39, 1], stride=[1, 1], padding=[19, 0], dilation=[1, 1], groups=256, bias=False) #Model::Model/Conv2d[n_Conv_66]/input.42
        self.module_68 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_67]/9433
        self.module_69 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_68]/9452
        self.module_70 = py_nndct.nn.Add() #Model::Model/t_734
        self.module_71 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.43
        self.module_72 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[39, 1], stride=[1, 1], padding=[19, 0], dilation=[1, 1], groups=256, bias=False) #Model::Model/Conv2d[n_Conv_71]/input.44
        self.module_73 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_72]/t_1074
        self.module_74 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.45
        self.module_75 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[39, 1], stride=[1, 1], padding=[19, 0], dilation=[1, 1], groups=256, bias=False) #Model::Model/Conv2d[n_Conv_74]/input.46
        self.module_76 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_75]/t_1077
        self.module_77 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.47
        self.module_78 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[39, 1], stride=[1, 1], padding=[19, 0], dilation=[1, 1], groups=256, bias=False) #Model::Model/Conv2d[n_Conv_77]/input.48
        self.module_79 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_78]/t_1080
        self.module_80 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.49
        self.module_81 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[39, 1], stride=[1, 1], padding=[19, 0], dilation=[1, 1], groups=256, bias=False) #Model::Model/Conv2d[n_Conv_80]/input.50
        self.module_82 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_81]/t_1083
        self.module_83 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.51
        self.module_84 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[39, 1], stride=[1, 1], padding=[19, 0], dilation=[1, 1], groups=256, bias=False) #Model::Model/Conv2d[n_Conv_83]/input.52
        self.module_85 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_84]/9654
        self.module_86 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_85]/9673
        self.module_87 = py_nndct.nn.Add() #Model::Model/t_757
        self.module_88 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.53
        self.module_89 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[39, 1], stride=[1, 1], padding=[19, 0], dilation=[1, 1], groups=256, bias=False) #Model::Model/Conv2d[n_Conv_88]/input.54
        self.module_90 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_89]/t_1092
        self.module_91 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.55
        self.module_92 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[39, 1], stride=[1, 1], padding=[19, 0], dilation=[1, 1], groups=256, bias=False) #Model::Model/Conv2d[n_Conv_91]/input.56
        self.module_93 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_92]/t_1095
        self.module_94 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.57
        self.module_95 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[39, 1], stride=[1, 1], padding=[19, 0], dilation=[1, 1], groups=256, bias=False) #Model::Model/Conv2d[n_Conv_94]/input.58
        self.module_96 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_95]/t_1098
        self.module_97 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.59
        self.module_98 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[39, 1], stride=[1, 1], padding=[19, 0], dilation=[1, 1], groups=256, bias=False) #Model::Model/Conv2d[n_Conv_97]/input.60
        self.module_99 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_98]/t_1101
        self.module_100 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.61
        self.module_101 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[39, 1], stride=[1, 1], padding=[19, 0], dilation=[1, 1], groups=256, bias=False) #Model::Model/Conv2d[n_Conv_100]/input.62
        self.module_102 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_101]/9875
        self.module_103 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_102]/9894
        self.module_104 = py_nndct.nn.Add() #Model::Model/t_780
        self.module_105 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.63
        self.module_106 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[51, 1], stride=[1, 1], padding=[25, 0], dilation=[1, 1], groups=256, bias=False) #Model::Model/Conv2d[n_Conv_105]/input.64
        self.module_107 = py_nndct.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_106]/t_1110
        self.module_108 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.65
        self.module_109 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[51, 1], stride=[1, 1], padding=[25, 0], dilation=[1, 1], groups=512, bias=False) #Model::Model/Conv2d[n_Conv_108]/input.66
        self.module_110 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_109]/t_1113
        self.module_111 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.67
        self.module_112 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[51, 1], stride=[1, 1], padding=[25, 0], dilation=[1, 1], groups=512, bias=False) #Model::Model/Conv2d[n_Conv_111]/input.68
        self.module_113 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_112]/t_1116
        self.module_114 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.69
        self.module_115 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[51, 1], stride=[1, 1], padding=[25, 0], dilation=[1, 1], groups=512, bias=False) #Model::Model/Conv2d[n_Conv_114]/input.70
        self.module_116 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_115]/t_1119
        self.module_117 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.71
        self.module_118 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[51, 1], stride=[1, 1], padding=[25, 0], dilation=[1, 1], groups=512, bias=False) #Model::Model/Conv2d[n_Conv_117]/input.72
        self.module_119 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_118]/10096
        self.module_120 = py_nndct.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_119]/10115
        self.module_121 = py_nndct.nn.Add() #Model::Model/t_803
        self.module_122 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.73
        self.module_123 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[51, 1], stride=[1, 1], padding=[25, 0], dilation=[1, 1], groups=512, bias=False) #Model::Model/Conv2d[n_Conv_122]/input.74
        self.module_124 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_123]/t_1128
        self.module_125 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.75
        self.module_126 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[51, 1], stride=[1, 1], padding=[25, 0], dilation=[1, 1], groups=512, bias=False) #Model::Model/Conv2d[n_Conv_125]/input.76
        self.module_127 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_126]/t_1131
        self.module_128 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.77
        self.module_129 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[51, 1], stride=[1, 1], padding=[25, 0], dilation=[1, 1], groups=512, bias=False) #Model::Model/Conv2d[n_Conv_128]/input.78
        self.module_130 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_129]/t_1134
        self.module_131 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.79
        self.module_132 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[51, 1], stride=[1, 1], padding=[25, 0], dilation=[1, 1], groups=512, bias=False) #Model::Model/Conv2d[n_Conv_131]/input.80
        self.module_133 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_132]/t_1137
        self.module_134 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.81
        self.module_135 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[51, 1], stride=[1, 1], padding=[25, 0], dilation=[1, 1], groups=512, bias=False) #Model::Model/Conv2d[n_Conv_134]/input.82
        self.module_136 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_135]/10317
        self.module_137 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_136]/10336
        self.module_138 = py_nndct.nn.Add() #Model::Model/t_826
        self.module_139 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.83
        self.module_140 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[51, 1], stride=[1, 1], padding=[25, 0], dilation=[1, 1], groups=512, bias=False) #Model::Model/Conv2d[n_Conv_139]/input.84
        self.module_141 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_140]/t_1146
        self.module_142 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.85
        self.module_143 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[51, 1], stride=[1, 1], padding=[25, 0], dilation=[1, 1], groups=512, bias=False) #Model::Model/Conv2d[n_Conv_142]/input.86
        self.module_144 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_143]/t_1149
        self.module_145 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.87
        self.module_146 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[51, 1], stride=[1, 1], padding=[25, 0], dilation=[1, 1], groups=512, bias=False) #Model::Model/Conv2d[n_Conv_145]/input.88
        self.module_147 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_146]/t_1152
        self.module_148 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.89
        self.module_149 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[51, 1], stride=[1, 1], padding=[25, 0], dilation=[1, 1], groups=512, bias=False) #Model::Model/Conv2d[n_Conv_148]/input.90
        self.module_150 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_149]/t_1155
        self.module_151 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.91
        self.module_152 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[51, 1], stride=[1, 1], padding=[25, 0], dilation=[1, 1], groups=512, bias=False) #Model::Model/Conv2d[n_Conv_151]/input.92
        self.module_153 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_152]/10538
        self.module_154 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_153]/10557
        self.module_155 = py_nndct.nn.Add() #Model::Model/t_849
        self.module_156 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.93
        self.module_157 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[63, 1], stride=[1, 1], padding=[31, 0], dilation=[1, 1], groups=512, bias=False) #Model::Model/Conv2d[n_Conv_156]/input.94
        self.module_158 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_157]/t_1164
        self.module_159 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.95
        self.module_160 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[63, 1], stride=[1, 1], padding=[31, 0], dilation=[1, 1], groups=512, bias=False) #Model::Model/Conv2d[n_Conv_159]/input.96
        self.module_161 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_160]/t_1167
        self.module_162 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.97
        self.module_163 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[63, 1], stride=[1, 1], padding=[31, 0], dilation=[1, 1], groups=512, bias=False) #Model::Model/Conv2d[n_Conv_162]/input.98
        self.module_164 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_163]/t_1170
        self.module_165 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.99
        self.module_166 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[63, 1], stride=[1, 1], padding=[31, 0], dilation=[1, 1], groups=512, bias=False) #Model::Model/Conv2d[n_Conv_165]/input.100
        self.module_167 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_166]/t_1173
        self.module_168 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.101
        self.module_169 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[63, 1], stride=[1, 1], padding=[31, 0], dilation=[1, 1], groups=512, bias=False) #Model::Model/Conv2d[n_Conv_168]/input.102
        self.module_170 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_169]/10759
        self.module_171 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_170]/10778
        self.module_172 = py_nndct.nn.Add() #Model::Model/t_872
        self.module_173 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.103
        self.module_174 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[63, 1], stride=[1, 1], padding=[31, 0], dilation=[1, 1], groups=512, bias=False) #Model::Model/Conv2d[n_Conv_173]/input.104
        self.module_175 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_174]/t_1182
        self.module_176 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.105
        self.module_177 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[63, 1], stride=[1, 1], padding=[31, 0], dilation=[1, 1], groups=512, bias=False) #Model::Model/Conv2d[n_Conv_176]/input.106
        self.module_178 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_177]/t_1185
        self.module_179 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.107
        self.module_180 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[63, 1], stride=[1, 1], padding=[31, 0], dilation=[1, 1], groups=512, bias=False) #Model::Model/Conv2d[n_Conv_179]/input.108
        self.module_181 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_180]/t_1188
        self.module_182 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.109
        self.module_183 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[63, 1], stride=[1, 1], padding=[31, 0], dilation=[1, 1], groups=512, bias=False) #Model::Model/Conv2d[n_Conv_182]/input.110
        self.module_184 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_183]/t_1191
        self.module_185 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.111
        self.module_186 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[63, 1], stride=[1, 1], padding=[31, 0], dilation=[1, 1], groups=512, bias=False) #Model::Model/Conv2d[n_Conv_185]/input.112
        self.module_187 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_186]/10980
        self.module_188 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_187]/10999
        self.module_189 = py_nndct.nn.Add() #Model::Model/t_895
        self.module_190 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.113
        self.module_191 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[63, 1], stride=[1, 1], padding=[31, 0], dilation=[1, 1], groups=512, bias=False) #Model::Model/Conv2d[n_Conv_190]/input.114
        self.module_192 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_191]/t_1200
        self.module_193 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.115
        self.module_194 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[63, 1], stride=[1, 1], padding=[31, 0], dilation=[1, 1], groups=512, bias=False) #Model::Model/Conv2d[n_Conv_193]/input.116
        self.module_195 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_194]/t_1203
        self.module_196 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.117
        self.module_197 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[63, 1], stride=[1, 1], padding=[31, 0], dilation=[1, 1], groups=512, bias=False) #Model::Model/Conv2d[n_Conv_196]/input.118
        self.module_198 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_197]/t_1206
        self.module_199 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.119
        self.module_200 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[63, 1], stride=[1, 1], padding=[31, 0], dilation=[1, 1], groups=512, bias=False) #Model::Model/Conv2d[n_Conv_199]/input.120
        self.module_201 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_200]/t_1209
        self.module_202 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.121
        self.module_203 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[63, 1], stride=[1, 1], padding=[31, 0], dilation=[1, 1], groups=512, bias=False) #Model::Model/Conv2d[n_Conv_202]/input.122
        self.module_204 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_203]/11201
        self.module_205 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_204]/11220
        self.module_206 = py_nndct.nn.Add() #Model::Model/t_918
        self.module_207 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.123
        self.module_208 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[75, 1], stride=[1, 1], padding=[37, 0], dilation=[1, 1], groups=512, bias=False) #Model::Model/Conv2d[n_Conv_207]/input.124
        self.module_209 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_208]/t_1218
        self.module_210 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.125
        self.module_211 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[75, 1], stride=[1, 1], padding=[37, 0], dilation=[1, 1], groups=512, bias=False) #Model::Model/Conv2d[n_Conv_210]/input.126
        self.module_212 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_211]/t_1221
        self.module_213 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.127
        self.module_214 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[75, 1], stride=[1, 1], padding=[37, 0], dilation=[1, 1], groups=512, bias=False) #Model::Model/Conv2d[n_Conv_213]/input.128
        self.module_215 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_214]/t_1224
        self.module_216 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.129
        self.module_217 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[75, 1], stride=[1, 1], padding=[37, 0], dilation=[1, 1], groups=512, bias=False) #Model::Model/Conv2d[n_Conv_216]/input.130
        self.module_218 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_217]/t_1227
        self.module_219 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.131
        self.module_220 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[75, 1], stride=[1, 1], padding=[37, 0], dilation=[1, 1], groups=512, bias=False) #Model::Model/Conv2d[n_Conv_219]/input.132
        self.module_221 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_220]/11422
        self.module_222 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_221]/11441
        self.module_223 = py_nndct.nn.Add() #Model::Model/t_941
        self.module_224 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.133
        self.module_225 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[75, 1], stride=[1, 1], padding=[37, 0], dilation=[1, 1], groups=512, bias=False) #Model::Model/Conv2d[n_Conv_224]/input.134
        self.module_226 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_225]/t_1236
        self.module_227 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.135
        self.module_228 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[75, 1], stride=[1, 1], padding=[37, 0], dilation=[1, 1], groups=512, bias=False) #Model::Model/Conv2d[n_Conv_227]/input.136
        self.module_229 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_228]/t_1239
        self.module_230 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.137
        self.module_231 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[75, 1], stride=[1, 1], padding=[37, 0], dilation=[1, 1], groups=512, bias=False) #Model::Model/Conv2d[n_Conv_230]/input.138
        self.module_232 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_231]/t_1242
        self.module_233 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.139
        self.module_234 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[75, 1], stride=[1, 1], padding=[37, 0], dilation=[1, 1], groups=512, bias=False) #Model::Model/Conv2d[n_Conv_233]/input.140
        self.module_235 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_234]/t_1245
        self.module_236 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.141
        self.module_237 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[75, 1], stride=[1, 1], padding=[37, 0], dilation=[1, 1], groups=512, bias=False) #Model::Model/Conv2d[n_Conv_236]/input.142
        self.module_238 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_237]/11643
        self.module_239 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_238]/11662
        self.module_240 = py_nndct.nn.Add() #Model::Model/t_964
        self.module_241 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.143
        self.module_242 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[75, 1], stride=[1, 1], padding=[37, 0], dilation=[1, 1], groups=512, bias=False) #Model::Model/Conv2d[n_Conv_241]/input.144
        self.module_243 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_242]/t_1254
        self.module_244 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.145
        self.module_245 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[75, 1], stride=[1, 1], padding=[37, 0], dilation=[1, 1], groups=512, bias=False) #Model::Model/Conv2d[n_Conv_244]/input.146
        self.module_246 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_245]/t_1257
        self.module_247 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.147
        self.module_248 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[75, 1], stride=[1, 1], padding=[37, 0], dilation=[1, 1], groups=512, bias=False) #Model::Model/Conv2d[n_Conv_247]/input.148
        self.module_249 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_248]/t_1260
        self.module_250 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.149
        self.module_251 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[75, 1], stride=[1, 1], padding=[37, 0], dilation=[1, 1], groups=512, bias=False) #Model::Model/Conv2d[n_Conv_250]/input.150
        self.module_252 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_251]/t_1263
        self.module_253 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.151
        self.module_254 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[75, 1], stride=[1, 1], padding=[37, 0], dilation=[1, 1], groups=512, bias=False) #Model::Model/Conv2d[n_Conv_253]/input.152
        self.module_255 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_254]/11864
        self.module_256 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_255]/11883
        self.module_257 = py_nndct.nn.Add() #Model::Model/t_987
        self.module_258 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.153
        self.module_259 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[87, 1], stride=[1, 1], padding=[86, 0], dilation=[2, 2], groups=512, bias=False) #Model::Model/Conv2d[n_Conv_258]/input.154
        self.module_260 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_259]/t_1272
        self.module_261 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input.155
        self.module_262 = py_nndct.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_261]/t_1275
        self.module_263 = py_nndct.nn.ReLU(inplace=False) #Model::Model/input
        self.module_264 = py_nndct.nn.Conv2d(in_channels=1024, out_channels=29, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #Model::Model/Conv2d[n_Conv_263]/11965
        self.module_265 = py_nndct.nn.Module('squeeze') #Model::Model/11967
        self.module_266 = py_nndct.nn.Module('permute') #Model::Model/11972

    def forward(self, *args):
        output_module_0 = self.module_0(input=args[0])
        output_module_0 = self.module_1(output_module_0)
        output_module_0 = self.module_2(output_module_0)
        output_module_0 = self.module_3(output_module_0)
        output_module_4 = self.module_4(output_module_0)
        output_module_4 = self.module_5(output_module_4)
        output_module_4 = self.module_6(output_module_4)
        output_module_4 = self.module_7(output_module_4)
        output_module_4 = self.module_8(output_module_4)
        output_module_4 = self.module_9(output_module_4)
        output_module_4 = self.module_10(output_module_4)
        output_module_4 = self.module_11(output_module_4)
        output_module_4 = self.module_12(output_module_4)
        output_module_4 = self.module_13(output_module_4)
        output_module_4 = self.module_14(output_module_4)
        output_module_4 = self.module_15(output_module_4)
        output_module_4 = self.module_16(output_module_4)
        output_module_4 = self.module_17(output_module_4)
        output_module_18 = self.module_18(output_module_0)
        output_module_4 = self.module_19(input=output_module_4, other=output_module_18, alpha=1)
        output_module_4 = self.module_20(output_module_4)
        output_module_21 = self.module_21(output_module_4)
        output_module_21 = self.module_22(output_module_21)
        output_module_21 = self.module_23(output_module_21)
        output_module_21 = self.module_24(output_module_21)
        output_module_21 = self.module_25(output_module_21)
        output_module_21 = self.module_26(output_module_21)
        output_module_21 = self.module_27(output_module_21)
        output_module_21 = self.module_28(output_module_21)
        output_module_21 = self.module_29(output_module_21)
        output_module_21 = self.module_30(output_module_21)
        output_module_21 = self.module_31(output_module_21)
        output_module_21 = self.module_32(output_module_21)
        output_module_21 = self.module_33(output_module_21)
        output_module_21 = self.module_34(output_module_21)
        output_module_35 = self.module_35(output_module_4)
        output_module_21 = self.module_36(input=output_module_21, other=output_module_35, alpha=1)
        output_module_21 = self.module_37(output_module_21)
        output_module_38 = self.module_38(output_module_21)
        output_module_38 = self.module_39(output_module_38)
        output_module_38 = self.module_40(output_module_38)
        output_module_38 = self.module_41(output_module_38)
        output_module_38 = self.module_42(output_module_38)
        output_module_38 = self.module_43(output_module_38)
        output_module_38 = self.module_44(output_module_38)
        output_module_38 = self.module_45(output_module_38)
        output_module_38 = self.module_46(output_module_38)
        output_module_38 = self.module_47(output_module_38)
        output_module_38 = self.module_48(output_module_38)
        output_module_38 = self.module_49(output_module_38)
        output_module_38 = self.module_50(output_module_38)
        output_module_38 = self.module_51(output_module_38)
        output_module_52 = self.module_52(output_module_21)
        output_module_38 = self.module_53(input=output_module_38, other=output_module_52, alpha=1)
        output_module_38 = self.module_54(output_module_38)
        output_module_55 = self.module_55(output_module_38)
        output_module_55 = self.module_56(output_module_55)
        output_module_55 = self.module_57(output_module_55)
        output_module_55 = self.module_58(output_module_55)
        output_module_55 = self.module_59(output_module_55)
        output_module_55 = self.module_60(output_module_55)
        output_module_55 = self.module_61(output_module_55)
        output_module_55 = self.module_62(output_module_55)
        output_module_55 = self.module_63(output_module_55)
        output_module_55 = self.module_64(output_module_55)
        output_module_55 = self.module_65(output_module_55)
        output_module_55 = self.module_66(output_module_55)
        output_module_55 = self.module_67(output_module_55)
        output_module_55 = self.module_68(output_module_55)
        output_module_69 = self.module_69(output_module_38)
        output_module_55 = self.module_70(input=output_module_55, other=output_module_69, alpha=1)
        output_module_55 = self.module_71(output_module_55)
        output_module_72 = self.module_72(output_module_55)
        output_module_72 = self.module_73(output_module_72)
        output_module_72 = self.module_74(output_module_72)
        output_module_72 = self.module_75(output_module_72)
        output_module_72 = self.module_76(output_module_72)
        output_module_72 = self.module_77(output_module_72)
        output_module_72 = self.module_78(output_module_72)
        output_module_72 = self.module_79(output_module_72)
        output_module_72 = self.module_80(output_module_72)
        output_module_72 = self.module_81(output_module_72)
        output_module_72 = self.module_82(output_module_72)
        output_module_72 = self.module_83(output_module_72)
        output_module_72 = self.module_84(output_module_72)
        output_module_72 = self.module_85(output_module_72)
        output_module_86 = self.module_86(output_module_55)
        output_module_72 = self.module_87(input=output_module_72, other=output_module_86, alpha=1)
        output_module_72 = self.module_88(output_module_72)
        output_module_89 = self.module_89(output_module_72)
        output_module_89 = self.module_90(output_module_89)
        output_module_89 = self.module_91(output_module_89)
        output_module_89 = self.module_92(output_module_89)
        output_module_89 = self.module_93(output_module_89)
        output_module_89 = self.module_94(output_module_89)
        output_module_89 = self.module_95(output_module_89)
        output_module_89 = self.module_96(output_module_89)
        output_module_89 = self.module_97(output_module_89)
        output_module_89 = self.module_98(output_module_89)
        output_module_89 = self.module_99(output_module_89)
        output_module_89 = self.module_100(output_module_89)
        output_module_89 = self.module_101(output_module_89)
        output_module_89 = self.module_102(output_module_89)
        output_module_103 = self.module_103(output_module_72)
        output_module_89 = self.module_104(input=output_module_89, other=output_module_103, alpha=1)
        output_module_89 = self.module_105(output_module_89)
        output_module_106 = self.module_106(output_module_89)
        output_module_106 = self.module_107(output_module_106)
        output_module_106 = self.module_108(output_module_106)
        output_module_106 = self.module_109(output_module_106)
        output_module_106 = self.module_110(output_module_106)
        output_module_106 = self.module_111(output_module_106)
        output_module_106 = self.module_112(output_module_106)
        output_module_106 = self.module_113(output_module_106)
        output_module_106 = self.module_114(output_module_106)
        output_module_106 = self.module_115(output_module_106)
        output_module_106 = self.module_116(output_module_106)
        output_module_106 = self.module_117(output_module_106)
        output_module_106 = self.module_118(output_module_106)
        output_module_106 = self.module_119(output_module_106)
        output_module_120 = self.module_120(output_module_89)
        output_module_106 = self.module_121(input=output_module_106, other=output_module_120, alpha=1)
        output_module_106 = self.module_122(output_module_106)
        output_module_123 = self.module_123(output_module_106)
        output_module_123 = self.module_124(output_module_123)
        output_module_123 = self.module_125(output_module_123)
        output_module_123 = self.module_126(output_module_123)
        output_module_123 = self.module_127(output_module_123)
        output_module_123 = self.module_128(output_module_123)
        output_module_123 = self.module_129(output_module_123)
        output_module_123 = self.module_130(output_module_123)
        output_module_123 = self.module_131(output_module_123)
        output_module_123 = self.module_132(output_module_123)
        output_module_123 = self.module_133(output_module_123)
        output_module_123 = self.module_134(output_module_123)
        output_module_123 = self.module_135(output_module_123)
        output_module_123 = self.module_136(output_module_123)
        output_module_137 = self.module_137(output_module_106)
        output_module_123 = self.module_138(input=output_module_123, other=output_module_137, alpha=1)
        output_module_123 = self.module_139(output_module_123)
        output_module_140 = self.module_140(output_module_123)
        output_module_140 = self.module_141(output_module_140)
        output_module_140 = self.module_142(output_module_140)
        output_module_140 = self.module_143(output_module_140)
        output_module_140 = self.module_144(output_module_140)
        output_module_140 = self.module_145(output_module_140)
        output_module_140 = self.module_146(output_module_140)
        output_module_140 = self.module_147(output_module_140)
        output_module_140 = self.module_148(output_module_140)
        output_module_140 = self.module_149(output_module_140)
        output_module_140 = self.module_150(output_module_140)
        output_module_140 = self.module_151(output_module_140)
        output_module_140 = self.module_152(output_module_140)
        output_module_140 = self.module_153(output_module_140)
        output_module_154 = self.module_154(output_module_123)
        output_module_140 = self.module_155(input=output_module_140, other=output_module_154, alpha=1)
        output_module_140 = self.module_156(output_module_140)
        output_module_157 = self.module_157(output_module_140)
        output_module_157 = self.module_158(output_module_157)
        output_module_157 = self.module_159(output_module_157)
        output_module_157 = self.module_160(output_module_157)
        output_module_157 = self.module_161(output_module_157)
        output_module_157 = self.module_162(output_module_157)
        output_module_157 = self.module_163(output_module_157)
        output_module_157 = self.module_164(output_module_157)
        output_module_157 = self.module_165(output_module_157)
        output_module_157 = self.module_166(output_module_157)
        output_module_157 = self.module_167(output_module_157)
        output_module_157 = self.module_168(output_module_157)
        output_module_157 = self.module_169(output_module_157)
        output_module_157 = self.module_170(output_module_157)
        output_module_171 = self.module_171(output_module_140)
        output_module_157 = self.module_172(input=output_module_157, other=output_module_171, alpha=1)
        output_module_157 = self.module_173(output_module_157)
        output_module_174 = self.module_174(output_module_157)
        output_module_174 = self.module_175(output_module_174)
        output_module_174 = self.module_176(output_module_174)
        output_module_174 = self.module_177(output_module_174)
        output_module_174 = self.module_178(output_module_174)
        output_module_174 = self.module_179(output_module_174)
        output_module_174 = self.module_180(output_module_174)
        output_module_174 = self.module_181(output_module_174)
        output_module_174 = self.module_182(output_module_174)
        output_module_174 = self.module_183(output_module_174)
        output_module_174 = self.module_184(output_module_174)
        output_module_174 = self.module_185(output_module_174)
        output_module_174 = self.module_186(output_module_174)
        output_module_174 = self.module_187(output_module_174)
        output_module_188 = self.module_188(output_module_157)
        output_module_174 = self.module_189(input=output_module_174, other=output_module_188, alpha=1)
        output_module_174 = self.module_190(output_module_174)
        output_module_191 = self.module_191(output_module_174)
        output_module_191 = self.module_192(output_module_191)
        output_module_191 = self.module_193(output_module_191)
        output_module_191 = self.module_194(output_module_191)
        output_module_191 = self.module_195(output_module_191)
        output_module_191 = self.module_196(output_module_191)
        output_module_191 = self.module_197(output_module_191)
        output_module_191 = self.module_198(output_module_191)
        output_module_191 = self.module_199(output_module_191)
        output_module_191 = self.module_200(output_module_191)
        output_module_191 = self.module_201(output_module_191)
        output_module_191 = self.module_202(output_module_191)
        output_module_191 = self.module_203(output_module_191)
        output_module_191 = self.module_204(output_module_191)
        output_module_205 = self.module_205(output_module_174)
        output_module_191 = self.module_206(input=output_module_191, other=output_module_205, alpha=1)
        output_module_191 = self.module_207(output_module_191)
        output_module_208 = self.module_208(output_module_191)
        output_module_208 = self.module_209(output_module_208)
        output_module_208 = self.module_210(output_module_208)
        output_module_208 = self.module_211(output_module_208)
        output_module_208 = self.module_212(output_module_208)
        output_module_208 = self.module_213(output_module_208)
        output_module_208 = self.module_214(output_module_208)
        output_module_208 = self.module_215(output_module_208)
        output_module_208 = self.module_216(output_module_208)
        output_module_208 = self.module_217(output_module_208)
        output_module_208 = self.module_218(output_module_208)
        output_module_208 = self.module_219(output_module_208)
        output_module_208 = self.module_220(output_module_208)
        output_module_208 = self.module_221(output_module_208)
        output_module_222 = self.module_222(output_module_191)
        output_module_208 = self.module_223(input=output_module_208, other=output_module_222, alpha=1)
        output_module_208 = self.module_224(output_module_208)
        output_module_225 = self.module_225(output_module_208)
        output_module_225 = self.module_226(output_module_225)
        output_module_225 = self.module_227(output_module_225)
        output_module_225 = self.module_228(output_module_225)
        output_module_225 = self.module_229(output_module_225)
        output_module_225 = self.module_230(output_module_225)
        output_module_225 = self.module_231(output_module_225)
        output_module_225 = self.module_232(output_module_225)
        output_module_225 = self.module_233(output_module_225)
        output_module_225 = self.module_234(output_module_225)
        output_module_225 = self.module_235(output_module_225)
        output_module_225 = self.module_236(output_module_225)
        output_module_225 = self.module_237(output_module_225)
        output_module_225 = self.module_238(output_module_225)
        output_module_239 = self.module_239(output_module_208)
        output_module_225 = self.module_240(input=output_module_225, other=output_module_239, alpha=1)
        output_module_225 = self.module_241(output_module_225)
        output_module_242 = self.module_242(output_module_225)
        output_module_242 = self.module_243(output_module_242)
        output_module_242 = self.module_244(output_module_242)
        output_module_242 = self.module_245(output_module_242)
        output_module_242 = self.module_246(output_module_242)
        output_module_242 = self.module_247(output_module_242)
        output_module_242 = self.module_248(output_module_242)
        output_module_242 = self.module_249(output_module_242)
        output_module_242 = self.module_250(output_module_242)
        output_module_242 = self.module_251(output_module_242)
        output_module_242 = self.module_252(output_module_242)
        output_module_242 = self.module_253(output_module_242)
        output_module_242 = self.module_254(output_module_242)
        output_module_242 = self.module_255(output_module_242)
        output_module_256 = self.module_256(output_module_225)
        output_module_242 = self.module_257(input=output_module_242, other=output_module_256, alpha=1)
        output_module_242 = self.module_258(output_module_242)
        output_module_242 = self.module_259(output_module_242)
        output_module_242 = self.module_260(output_module_242)
        output_module_242 = self.module_261(output_module_242)
        output_module_242 = self.module_262(output_module_242)
        output_module_242 = self.module_263(output_module_242)
        output_module_242 = self.module_264(output_module_242)
        output_module_242 = self.module_265(input=output_module_242, dim=(3))
        output_module_242 = self.module_266(dims=[0,2,1], input=output_module_242)
        return output_module_242
