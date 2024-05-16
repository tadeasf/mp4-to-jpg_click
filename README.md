# Video Frame Extraction Tool

This tool allows you to extract frames from video files, with features like deduplication and dynamic frame skipping.

## Features

This Video Frame Extraction Tool is designed to optimize your workflow with powerful, customizable features for handling video frame processing. Below are the standout capabilities:

+ **Versatile Video Format Support:** Work with popular video formats including .mp4 and .mov.
+ **Frame Deduplication:** Integrates an intelligent algorithm to eliminate similar frames, ensuring each extracted frame is unique. Customize the similarity threshold to fit your needs.
+ **Adaptive Frame Skipping:** Dynamically skips frames to maintain efficiency without sacrificing quality. The tool automatically adjusts skipping based on the video's FPS and duration, or you can customize the settings to your preferences.
+ **Custom Frame Limits:** Exercise complete control over your output. Set a cap on the maximum number of frames extracted from each video or across all your video processing tasks.
+ **Fast Processing:** Leverages the power of multi-core CPUs to speed up frame extraction, enabling parallel processing of video files.
+ **User-Friendly GUI:** A graphical user interface simplifies file selection and settings adjustments, making it accessible to users of all skill levels.


## Requirements

* Python 3.6+
* Required Python packages (listed in `requirements.txt`):
    * `opencv-python`
    * `tqdm`
    * `loguru`
    * `imagededup`
    * `click`
    * `tkinter`

## Installation

1. Clone the repository:

   ```bash
   git clone [https://github.com/yourusername/video-frame-extraction-tool.git](https://github.com/yourusername/video-frame-extraction-tool.git)
   cd video-frame-extraction-tool
   ```
2. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

Usage:

   ```bash
   python extract_frames.py [OPTIONS]
   ```

## CLI arguments:

Tailor the tool's operation with command-line arguments to fit your project requirements:

+ **--threshold FLOAT** Set the similarity threshold for deduplication (range: 0.5 to 1.0). A higher value results in stricter deduplication.
+ **--no-deduplication** Disables the frame deduplication feature.
+ **--no-fps-skip** Stops the tool from skipping frames based on FPS, extracting every frame.
+ **--no-time-skip** Prevents frame skipping based on video duration.
+ **--max-frames INTEGER** Limit the total number of frames to be generated from all processed videos.
+ **--max-frames-per-video INTEGER** Specify a maximum number of frames to be extracted from each individual video.


Example usage:
   ```bash
   python extract_frames.py --threshold 0.95 --max-frames 5000 --max-frames-per-video 1000
   ```

This fine-tuned approach provides a robust and flexible solution to efficiently manage frame extraction projects with precision control over output quality and operational parameters.
## How It Works

### Frame Extraction & Optimization

+ **Variable Frame Skipping:** Based on the set parameters and video characteristics (FPS and length), the tool calculates an optimal frame skipping strategy to balance between processing speed and output quality.
+ **Parallel Processing:** Distributes the workload across available CPU cores, significantly reducing the time required for frame extraction.
+ **Efficient Frame Saving:** Extracted frames are saved as JPG images, with an option to adjust the quality and resolution to meet specific needs.

### Deduplication for Quality Control:

+ Utilizes a CNN-based method to analyze and compare frames for near-duplicate detection.
+ Customizable threshold: Allows precision control over what is considered a duplicate, adjustable via the --threshold argument.
+ Organizes and relocates duplicates to a designated folder, ensuring your main output directory contains only unique frames.

### Contributing
Contributions are welcome! Please open issues or submit pull requests.

### Contact
For questions or feedback, please contact bussines@tadeasfort.com].

