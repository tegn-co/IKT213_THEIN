import cv2
import time


def save_camera_information(output_path="solutions/camera_outputs.txt"):

    capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not capture.isOpened():
        print("Could not open camera")
        return

    start = time.time()
    for _ in range(60):
        ret, frame = capture.read()
        if not ret:
            break
    end = time.time()
    seconds = end - start

    camera_fps = int(60 / seconds)
    camera_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    camera_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))


    with open(output_path, "w") as f:
        f.write(f"fps: {camera_fps}\n")
        f.write(f"width: {camera_width}\n")
        f.write(f"height: {camera_height}\n")

    capture.release()

    print("Saved camera information")

def main():
    save_camera_information()

main()