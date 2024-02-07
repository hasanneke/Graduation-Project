import cv2


def compress_video(filename):
    # Load the video
    cap = cv2.VideoCapture(f"./videos/{filename}")

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # Set lower resolution, for example, 640x480
    lower_resolution = (480, 854)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        f"./compressed_videos/{filename}", fourcc, 20.0, lower_resolution
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame
        resized_frame = cv2.resize(frame, lower_resolution)

        # Write the resized frame to output video
        out.write(resized_frame)

    # Release everything when done
    cap.release()
    out.release()
