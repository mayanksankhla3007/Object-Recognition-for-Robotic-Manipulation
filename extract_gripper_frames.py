import cv2
import os

# Paths
VIDEO_DIR = "gripper_video"
GRIPPER_FRAMES_DIR = "gripper_frames"
FRAME_STEP = 5  # Save every 5th frame

def create_folder(folder_path):
    os.makedirs(folder_path, exist_ok=True)

def extract_gripper_frames(video_path, output_folder, frame_step=5):
    create_folder(output_folder)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0

    if not cap.isOpened():
        print(f"❌ Failed to open video: {video_path}")
        return

    print(f"📂 Processing: {video_path}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save every Nth frame
        if frame_count % frame_step == 0:
            frame_filename = os.path.join(output_folder, f"frame_{saved_count:04d}.png")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"✅ Saved {saved_count} frames to: {output_folder}\n")

def extract_all_videos(video_dir, frames_dir):
    for file_name in os.listdir(video_dir):
        if file_name.endswith(".mp4"):
            session_name = os.path.splitext(file_name)[0]
            video_path = os.path.join(video_dir, file_name)
            output_folder = os.path.join(frames_dir, session_name)
            extract_gripper_frames(video_path, output_folder, frame_step=FRAME_STEP)

if __name__ == "__main__":
    create_folder(GRIPPER_FRAMES_DIR)
    extract_all_videos(VIDEO_DIR, GRIPPER_FRAMES_DIR)
