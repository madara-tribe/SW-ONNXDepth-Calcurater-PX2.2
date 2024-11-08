import argparse
import cv2
import time
import sys
from yolo_onnx import YOLODetect


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame_interval', type=int, default=5, help='camera interval to reduce burden')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='conf threshold for NMS or postprocess')
    parser.add_argument('--max_disparity', type=int, default=1000, help='max disparity')
    parser.add_argument('--min_disparity', type=int, default=10, help='max disparity')
    parser.add_argument('-p', '--yolo_onnx_path', type=str, default='weights/yolov7Tiny_640_640.onnx', help='yolo onnx weight model')
    opt = parser.parse_args()
    return opt

def main(opt):
    cap = cv2.VideoCapture(0)
    yolo_onnx = YOLODetect(opt)
    while True:
        ret, frame = cap.read()
        start_time = time.time()
        frame_, x_, y_ = yolo_onnx.inference_(frame)
        print("Prediction took {:.2f} seconds".format(time.time() - start_time))
        print("yolo shape", frame_.shape)
        cv2.imshow('camera' , frame_)
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
