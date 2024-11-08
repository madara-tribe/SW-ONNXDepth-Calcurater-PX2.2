import argparse
import sys
import numpy as np
import cv2
import time
from onnx_utils import Inference


def draw_depth(depth_map, w, h):
    set_min_depth = 0
    set_max_depth = 100
    # Normalize estimated depth to color it
    min_depth = depth_map.min()
    max_depth = depth_map.max()
    min_depth = min_depth if min_depth < set_min_depth else min_depth
    max_depth = max_depth if max_depth > set_max_depth else max_depth

    print(min_depth, max_depth)
    norm_depth_map = 255 * (depth_map - min_depth) / (max_depth - min_depth)
    norm_depth_map = 255 - norm_depth_map

    # Normalize and color the image
    color_depth = cv2.applyColorMap(cv2.convertScaleAbs(norm_depth_map, 1),
                                    cv2.COLORMAP_JET)
    #print(min_depth, max_depth, color_depth.min(), color_depth.max())
    # Resize the depth map to match the input image shape
    return cv2.resize(color_depth, (w, h))


def main(opt):
    cap = cv2.VideoCapture(0)
    inference_class = Inference(opt.onnx_model_path)
    while True:
        ret, frame = cap.read()
        h, w, _ = frame.shape
        start_time = time.time()
        depth_img = inference_class.inference_(frame)
        print("Prediction took {:.2f} seconds".format(time.time() - start_time))
        print()
        colored_depth = draw_depth(depth_img, w, h)
        #output_norm = cv2.normalize(depth_img, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        #output_norm = normalize_depth(output_norm, bits=2)
        #colored_depth = cv2.applyColorMap((output_norm*255).astype(np.uint8), cv2.COLORMAP_MAGMA)
        cv2.imshow('camera' , colored_depth)
        key =cv2.waitKey(10)
        if key == 27:
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--onnx_model_path', type=str, default='weights/dpt_large_384.onnx', help='onnx midas weight model')
    opt = parser.parse_args()
    try:
        main(opt)
    except KeyboardInterrupt:
        sys.exit(1)
        raise
