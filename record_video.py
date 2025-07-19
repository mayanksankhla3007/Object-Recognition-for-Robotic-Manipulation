import cv2
import depthai as dai
import os
import time

def create_folder(base_dir="gripper_video"):
    os.makedirs(base_dir, exist_ok=True)
    return base_dir

def get_next_session_name(base_dir="video_dataset", prefix="gripper_video_"):
    # Get the highest numbered session in the directory
    existing_sessions = [name for name in os.listdir(base_dir) if name.startswith(prefix) and name.endswith(".mp4")]
    session_numbers = [int(name.replace(prefix, "").replace(".mp4", "")) for name in existing_sessions if name.replace(prefix, "").replace(".mp4", "").isdigit()]
    
    if session_numbers:
        next_number = max(session_numbers) + 1
    else:
        next_number = 1
    
    return f"{prefix}{next_number:02d}"


def record_video(fps=30, resolution=(1920, 1080), preview_size=(960, 540)):
    folder_path = create_folder()
    session_name = get_next_session_name(folder_path)
    video_path = os.path.join(folder_path, f"{session_name}.mp4")

    # Setup DepthAI pipeline
    pipeline = dai.Pipeline()
    cam = pipeline.createColorCamera()
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setVideoSize(resolution)
    cam.setFps(fps)

    xout = pipeline.createXLinkOut()
    xout.setStreamName("video")
    cam.video.link(xout.input)

    with dai.Device(pipeline) as device:
        video_queue = device.getOutputQueue("video", maxSize=4, blocking=False)

        print("\n⏳ Starting in 5 seconds...")
        for i in range(5, 0, -1):
            print(f"\n⏳ Starting in {i} seconds...")
            time.sleep(1)

        print(f"Recording session '{session_name}' — press 'q' to stop.")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, fps, resolution)

        while True:
            in_frame = video_queue.get()
            frame = in_frame.getCvFrame()

            out.write(frame)

            # Resize for display only
            preview = cv2.resize(frame, preview_size)
            cv2.imshow("Recording (Press 'q' to stop)", preview)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Recording stopped by user.")
                break

        out.release()
        cv2.destroyAllWindows()
        print(f"Video saved at: {video_path}")

if __name__ == "__main__":
    print("OAK-Robotic Manipulation Video Recorder\n")
    record_video()
