import argparse
import cv2
import time
import sys
import numpy as np
from onnx_utils import YOLODetect, MidasDepth


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame_interval', type=int, default=5, help='camera interval to reduce burden')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='conf threshold for NMS or postprocess')
    parser.add_argument('--max_depth', type=int, default=50, help='max disparity 1 or 50 or 300')
    parser.add_argument('--min_depth', type=int, default=0, help='max disparity')
    parser.add_argument('-yp', '--yolo_onnx_path', type=str, default='weights/yolov7Tiny_640_640.onnx', help='yolo onnx weight model')
    parser.add_argument('-dp', '--depth_onnx_path', type=str, default='weights/dpt_large_384.onnx', help='depth onnx weight model')
    opt = parser.parse_args()
    return opt

def main(opt):
    cap = cv2.VideoCapture(0)
    yolo_onnx = YOLODetect(opt)
    midas_onnx = MidasDepth(opt)
    while True:
        ret, frame = cap.read()
        start_time = time.time()
        h, w, _ = frame.shape
        depth_img = midas_onnx.inference_(frame)
        colored_depth = midas_onnx.draw_depth(opt, depth_img, w, h)
        ob_depth, median_depth = yolo_onnx.inference_(frame, colored_depth)
        print("Prediction took {:.2f} seconds".format(time.time() - start_time))
        print('yolo shape, depth shape', median_depth, ob_depth.shape, colored_depth.shape)
        
        cv2.imshow('camera', ob_depth)
        key =cv2.waitKey(10)
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    opt = get_parser()
    try:
        main(opt)
    except KeyboardInterrupt:
        sys.exit(1)
        raise
