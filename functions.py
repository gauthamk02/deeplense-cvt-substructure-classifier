import numpy as np
import onnxruntime as rt

onnx_path = 'model/model.onnx'

def predict(img):
    session = rt.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    img = np.array(img).astype(np.float32)
    img = img.reshape(1, 1, 256, 256)
    img = img / 255.0
    pred = session.run([output_name], {input_name: img})[0]
    pred = np.exp(pred) / np.sum(np.exp(pred), axis=1, keepdims=True)

    class_probs = {'No Substructure': str(pred[0][0]), 'Substructure': str(pred[0][1])}
    return class_probs