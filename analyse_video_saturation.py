import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import argparse

def get_mean_saturation(img: np.ndarray) -> np.float32:
    '''
    Count mean saturation of an image
        Args:
            img: np.ndarray - image to count mean saturation
    '''
    image_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img32 = image_hsv.astype(np.float32)
    mean_saturation = np.mean(img32[:, :, 1], dtype=np.float32)
    return mean_saturation

def find_working_camera(start_device_id:int=0) -> cv2.VideoCapture:
    '''
    Find next working camera (up to 999 devices)
        Args:
            start_device_id: int - offset to select new online camera
        Return:
            cv2.VideoCapture if online camera found else None
    '''
    if type(start_device_id) != int:
        raise TypeError("Wrong type of start_device_id")
    for i in range(start_device_id, 1000):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            return cap
    return None

def process_video(source:cv2.VideoCapture):
    '''
    Process every frame of video
    Args:
        source: cv2.VideoCapture
    Return:
        timestamps: [int] - ms from video starts
        mean_saturations: [np.float32] - mean saturations of every frame by a timestamp
    '''
    if type(source) != cv2.VideoCapture:
        raise TypeError("source is not cv2.VideoCapture")
    timestamps = []
    mean_saturations = []
    start_time = int(time.time_ns())

    while True:
        ret, frame = source.read()
        key = cv2.waitKey(1)
        cv2.imshow('Frame', frame)
        timestamps.append(int(time.time_ns()) - start_time)
        mean_saturations.append(get_mean_saturation(frame))
        if key == 27: #Esc
            break

    cv2.destroyWindow('Frame')
    source.release()
    return timestamps, mean_saturations


def display_and_save_graph(x, y, path_to_save_graph=None) -> None:
    '''
    Display graph on screen and save to file
    Args:
        x - data for x-axis
        y - data for y-axis
        path_to_save_graph:str - path where graph should be saved
    '''
    plt.plot(x, y)
    if path_to_save_graph is not None:
        plt.savefig(path_to_save_graph)
    plt.show()


def main(path_to_save_graph: str):
    camera = find_working_camera()
    if camera is None:
        raise Exception("Available camera was not found")
    timestamps, mean_saturation = process_video(camera)
    display_and_save_graph(timestamps, mean_saturation, path_to_save_graph)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-path',type=str)
    args=parser.parse_args()
    main(args.path)