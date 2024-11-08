import cv2
import numpy as np
import onnxruntime
from midas_utils import call_transform
from yolov7s.common import letterbox, preprocess, onnx_inference, post_process

cuda = False

class MidasDepth:
    def __init__(self, opt):
        self.transform, self.net_h, self.net_w = call_transform()
        self.sess = onnxruntime.InferenceSession(opt.depth_onnx_path)
        
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
        
    def draw_depth(self, opt, depth_map, w, h):
        # Normalize estimated depth to color it
        min_depth = depth_map.min()
        max_depth = depth_map.max()
        min_depth = min_depth if min_depth < opt.min_depth else min_depth
        max_depth = max_depth if max_depth > opt.max_depth else max_depth
        #print("minmax", depth_map.min(), depth_map.max())
        norm_depth_map = 255 * (depth_map - min_depth) / (max_depth - min_depth)
        norm_depth_map = 255 - norm_depth_map

        # Normalize and color the image
        color_depth = cv2.applyColorMap(cv2.convertScaleAbs(norm_depth_map, 1),
                                        cv2.COLORMAP_JET)
        #print(min_depth, max_depth, color_depth.min(), color_depth.max())
        # Resize the depth map to match the input image shape
        return cv2.resize(color_depth, (w, h))

    def inference_(self, image):
        img_input = self.transform({"image": self.preprocess(image)})["image"]
        input_name = self.sess.get_inputs()[0].name
        output_name = self.sess.get_outputs()[0].name
        onnx_output = self.sess.run([output_name], {input_name: img_input.reshape(1, 3, self.net_h, self.net_w).astype(np.float32)})[0]
        #print('input_shape, onnx_output', img_input.shape, onnx_output.shape)
        
        return onnx_output[0]
    

class YOLODetect:
    def __init__(self, opt):
        self.conf_thres = opt.conf_thres
        self.init_onnx_model(opt)
        
    def init_onnx_model(self, opt):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        self.session = onnxruntime.InferenceSession(opt.yolo_onnx_path, providers=providers)
        IN_IMAGE_H = self.session.get_inputs()[0].shape[2]
        IN_IMAGE_W = self.session.get_inputs()[0].shape[3]
        self.new_shape = (IN_IMAGE_W, IN_IMAGE_H)
       
    def inference_(self, frame, colored_depth):
        ori_images = [colored_depth.copy()]
        resized_image, ratio, dwdh = letterbox(frame, new_shape=self.new_shape, auto=False)
        input_tensor = preprocess(resized_image)
        outputs = onnx_inference(self.session, input_tensor)
        pred_output, median_depth = post_process(outputs, ori_images, ratio, dwdh, self.conf_thres)
        return pred_output[0], median_depth

    
    

    
