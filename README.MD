# File Organizational Structure

CS445_FINAL_PROJ
- CS445_FAST_NST
  - content
  - models
    - epoch_1_Fri_Dec__1_15-20-09_2023_100000.0_10000000000.0.model
    - epoch_1_Mon_Dec__4_10-25-52_2023_100000.0_10000000000.0.model
    - epoch_1_Sun_Dec__3_14-35-54_2023_100000.0_10000000000.0.model
  - output
  - venv
  - .gitignore
  - README.MD
  - stylze.py
  - train.py
  - transformer.py
  - utils.py
  - vgg.py
  - video.py
- FP_coloraware_runs
  - FProjRenderImages
  - images
  - FProj_coloraware.ipynb
  - FProjRenderImages.ipynb

# Running Neural Style Transfer (slower version)

1. Under FP_coloraware_runs directory

# Running Fast Neural Style Transfer:

1. Under the CS445_FAST_NST directory, save an .mp4 content file in the directory.
2. Go to the video.py file and update the VIDEO_PATH variable on line 13 to the file path of the video.
3. On line 108, update the key in the STYLE_MODEL_PATHS dictionary to the style model to apply. You can use either "mosaic", "starry", or "picasso"
4. Update the OUTPUT_VIDEO_PATH variable on line 14 to the name of the output video.
5. Click run. It will first convert the input video to frames and save all frames in the content folder. Once all frames are saved, the stylize function will load all the frames in the content folder and apply Fast NST and save the frames to the output folder. Once all frames are processed, an .MP4 output video will be generated from all the frames in the output folder.
6. If you want to run a Fast NST on one picture/ frame, you can just save the picture as a .jpg in the content folder (and make sure there are no other pictures that you don't want to generate). You can then comment out lines 108, 112, and 113 and only run the stylize function, which will NST the image and save the single image in the output folder.
7. Example Content videos and Output Videos can be found in this DropBox link: https://www.dropbox.com/scl/fo/0vlez3q6guwl6dmqgt4bm/h?rlkey=96yy83r4dswevzl6c38ayxcda&dl=0
8. Example videos can also be found on this YouTube Link: https://youtu.be/wKzhDzAvUT4

