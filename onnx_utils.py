import cv2
import numpy as np
import onnxruntime as rt
from utils import call_transform


class Inference:
    def __init__(self, onnx_model_path):
        self.transform, self.net_h, self.net_w = call_transform()
        self.sess = rt.InferenceSession(onnx_model_path)
        
    def preprocess(self, img):
        """Read image and output RGB image (0-1).

        Args:
            path (str): path to file

        Returns:
            array: RGB image (0-1)
        """
        #if img.ndim == 2:
         #   img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        
    def inference_(self, image):
        img_input = self.transform({"image": self.preprocess(image)})["image"]
        input_name = self.sess.get_inputs()[0].name
        output_name = self.sess.get_outputs()[0].name
        onnx_output = self.sess.run([output_name], {input_name: img_input.reshape(1, 3, self.net_h, self.net_w).astype(np.float32)})[0]
        
        print('input_shape, onnx_output', img_input.shape, onnx_output.shape)
        
        return onnx_output[0]
    


    
