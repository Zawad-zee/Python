Idea of Snappy - While working on a project where I had to watch videos and take snaps of the changes of the frames of the video, it was tedious started off with a small length video and realised what would happen if it was a bit longer. Came the idea of automation. Started prompting gpt and developed the code. With hours of prompting and torubleshooting made it to work.

there are 3 .py files 
1) video_snapshot_detector.py - will just take the snapshots
2) video_snapshot_to_ppt.py - will take the snapshots as well as generate a ppt file with corresponding timestamps ( just in case)
3) video_snaps_pp2.py - will take the snapshots as well as generate a ppt file 
Video Frame Snapshot Detector
Automatically captures snapshots when video frames change significantly. Perfect for presentations, tutorials, and animated content.

Quick Start
1. Install Python Requirements
bash
pip install opencv-python numpy
2. Run the Program
Copy your video file to this folder
Edit the video filename in the code (line 108):
python
   video_file = "your_video_name.mp4"  # Change this!
Run: python video_snapshot_detector.py
Settings You Can Adjust
In the main() function, you can customize these settings:

python
detector = VideoFrameDetector(
    video_path=video_file,
    output_dir="video_snapshots",    # Where to save images
    threshold=5.0,                   # Lower = more sensitive
    min_interval=0.1,                # Minimum seconds between snapshots
    buffer_time=2.0,                 # Wait time for animations (seconds)
    debug=True                       # Show detailed output
)
Sensitivity Settings
Use Case	Threshold	Min Interval	Buffer Time
Presentations with animations	5.0	0.1	2.0
Fast-changing content	3.0	0.1	1.0
Slow presentations	10.0	1.0	3.0
Maximum sensitivity	1.0	0.1	2.0
Output
Creates a folder called video_snapshots
Saves images as: snapshot_20250910_143052_frame150_t5.23s.jpg
Shows progress and statistics in the console
Troubleshooting
"No module named 'cv2'" error:

bash
pip install opencv-python
"File not found" error:

Make sure your video file is in the same folder
Check the spelling of your filename
Try using the full path: video_file = r"C:\full\path\to\video.mp4"
Too many/few snapshots:

Too many: Increase threshold (try 10.0)
Too few: Decrease threshold (try 2.0)
Missing animations: Increase buffer_time (try 3.0)
Supported Video Formats
MP4, AVI, MOV, MKV, WMV, FLV
Most common video formats work
Requirements
Python 3.7 or higher
OpenCV (pip install opencv-python)
NumPy (pip install numpy)
Example Usage
python
# For a presentation with slow transitions
detector = VideoFrameDetector(
    video_path="presentation.mp4",
    threshold=8.0,
    buffer_time=3.0
)

# For fast-paced content
detector = VideoFrameDetector(
    video_path="tutorial.mp4", 
    threshold=3.0,
    min_interval=0.5,
    buffer_time=1.0
)
Tips for Best Results
Start with default settings and adjust if needed
Use debug=True to see what's happening
For animated text: Lower threshold (3.0-5.0) and buffer time (2.0-3.0)
For slide presentations: Medium threshold (8.0-15.0)
Test with a short video first to find optimal settings
Created with Python and OpenCV

