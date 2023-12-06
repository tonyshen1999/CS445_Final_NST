
import cv2
import os
import numpy as np
from stylize import stylize

STYLE_MODEL_PATHS = {
                     "starry-relu3": "models/starry_relu3_epoch2.model",
                     "mosaic-relu3": "models/mosaic_relu3_epoch2.model",
                     "picasso-relu3": "models/picasso_relu3_epoch2.model",
                     "mosaic": "models/mosaic_relu2_epoch1.model",
                     "starry": "models/starry_relu2_epoch1.model",
                     "picasso": "models/picasso_relu2_epoch1.model"
                    }

CONTENT_PATH = "content/"
OUTPUT_PATH = "output/"
INPUT_VIDEO_PATH = "no_god.mp4" # update this to the content video to apply NST
OUTPUT_VIDEO_PATH = "no_god_output_starry_relu3.mp4"
STYLE_MODEL = STYLE_MODEL_PATHS["starry-relu3"]

# write a numpy array of images into mp4
def vidwrite_from_numpy(output_video_path, np_images, fps = 30):

    # Define the frame rate (fps) for the output video

    # Get the height, width, and channels from the first frame
    height, width, _ = np_images[0].shape

    # Define the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use other codecs, e.g., 'XVID'
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Write each frame to the video writer
    for frame in np_images:
        # Ensure the frame is in uint8 format
        frame = frame.astype(np.uint8)
        # Write the frame to the video
        video_writer.write(frame)

    # Release the video writer
    video_writer.release()


def video2imageFolder(input_file, output_path):
    '''
    Extracts the frames from an input video file
    and saves them as separate frames in an output directory.
    Input:
        input_file: Input video file.
        output_path: Output directorys.
    Output:
        None

    Source: From CS445 Project 5 Utils Folder
    '''

    cap = cv2.VideoCapture()
    cap.open(input_file)

    if not cap.isOpened():
        print("Failed to open input video")

    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    frame_idx = 0

    while frame_idx < frame_count:
        ret, frame = cap.read()

        if not ret:
            print ("Failed to get the frame {}".format(frame_idx))
            continue

        out_name = os.path.join(output_path, 'f{:04d}.jpg'.format(frame_idx+1))
        ret = cv2.imwrite(out_name, frame)
        if not ret:
            print ("Failed to write the frame {}".format(frame_idx))
            continue

        frame_idx += 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

# load image from directory_path and return np array of images
def load_images_np(directory_path):

    # Get a list of all files in the directory
    file_list = os.listdir(directory_path)

    # Filter only files with '.jpg' extension
    image_files = [file for file in file_list if file.endswith('.jpg')]
    # Sort the image files based on their filenames
    image_files.sort()

    # Initialize an empty list to store the images
    images = []

    # Loop through the sorted image files and load each image
    for image_file in image_files:
        image_path = os.path.join(directory_path, image_file)
        image = cv2.imread(image_path)

        # Optional: Convert BGR to RGB if needed
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        images.append(image)

    # Convert the list of images to a NumPy array
    np_images = np.array(images)

    return np_images


# video2imageFolder(INPUT_VIDEO_PATH, CONTENT_PATH)  # convert video to images and save all frames to content folder
print("running NST")
stylize(STYLE_MODEL, CONTENT_PATH, OUTPUT_PATH)  # stylize all images in content folder using selected model, and save to output
# Load frames and save as output mp4
frames = load_images_np(OUTPUT_PATH)
vidwrite_from_numpy(OUTPUT_VIDEO_PATH, frames)
