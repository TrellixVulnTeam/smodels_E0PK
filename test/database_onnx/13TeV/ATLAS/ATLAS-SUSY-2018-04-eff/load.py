#!/usr/bin/env python3

""" load onnx model and make an inference with onnxruntime """
    
import onnxruntime, numpy

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def load():
    ort_session = onnxruntime.InferenceSession("model.onnx")
    ort_inputs = ort_session.get_inputs()
    for x in ort_inputs:
        print ( "input", x.name, "type", x.type, "shape", x.shape )
    inp = numpy.array ( [[ 15., 15. ]], dtype=numpy.float32 )
    # ort_outs = ort_session.run(None, { ort_inputs[0].name: inp } )
    ort_outs = ort_session.run(None, { "dense_input": inp } )
    nll = float ( ort_outs[0][0][0] )
    print ( "outputs", nll )

load()
