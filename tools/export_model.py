import numpy as np


def export_from_nemo(path, name):
    import nemo
    import nemo.collections.asr as nemo_asr
    quartznet = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name=name)
    quartznet.trainning = False
    quartznet.export(path, onnx_opset_version=9)

if __name__ == "__main__":
    path = '../onnx_quartznet.onnx'
    name = 'QuartzNet15x5Base-En'    
    print('Converting nemo {} model to {}'.format(path, name))
    export_from_nemo(path, name)
    print('Successfully exported.')
