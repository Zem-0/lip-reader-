import os
import subprocess


def convert_all_videos(input_dir, output_dir):
    """
    Convert all videos in input_dir to MP4 format using ffmpeg and save them in output_dir.

    Parameters:
    - input_dir (str): Directory containing input videos.
    - output_dir (str): Directory where converted videos will be saved.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # List all files in input directory
    files = os.listdir(input_dir)

    # Iterate through each file in the directory
    for file in files:
        if file.endswith(".mpg") or file.endswith(".avi") or file.endswith(".mpeg"):
            # Construct input and output paths
            input_path = os.path.join(input_dir, file)
            output_path = os.path.join(output_dir, f"{os.path.splitext(file)[0]}.mp4")

            # Use ffmpeg to convert video
            cmd = f'ffmpeg -i "{input_path}" -vcodec libx264 "{output_path}" -y'
            subprocess.run(cmd, shell=True)


if __name__ == "__main__":
    input_directory = 'data\s1'
    output_directory = 'output'
    convert_all_videos(input_directory, output_directory)