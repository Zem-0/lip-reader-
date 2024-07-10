import streamlit as st
import tensorflow as tf
import cv2
import os 

# Example of vocabulary and lookup layers
vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)
def preprocess_video(video_path, target_shape=(75, 46, 140, 1)):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized_frame = cv2.resize(gray_frame, (target_shape[2], target_shape[1]))
        frames.append(resized_frame)

    cap.release()

    # Convert to numpy array and add channel dimension
    frames = np.array(frames)
    frames = np.expand_dims(frames, axis=-1)  # Add channel dimension (height, width, 1)

    # Adjust sequence length
    if frames.shape[0] < target_shape[0]:
        # If the video has fewer frames, pad with zeros
        padding = np.zeros((target_shape[0] - frames.shape[0], target_shape[1], target_shape[2], target_shape[3]))
        frames = np.vstack((frames, padding))
    else:
        # If the video has more frames, truncate it
        frames = frames[:target_shape[0]]

    return np.expand_dims(frames, axis=0)
def load_video(path:str) -> tf.Tensor: 
    cap = cv2.VideoCapture(path)
    frames = []
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))): 
        ret, frame = cap.read()
        frame = tf.image.rgb_to_grayscale(frame)
        frames.append(frame[190:236,80:220,:])
    cap.release()
    
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    return tf.cast((frames - mean), tf.float32) / std

def load_alignments(path:str) -> tf.Tensor: 
    try:
        with open(path, 'r') as f: 
            lines = f.readlines() 
        tokens = []
        for line in lines:
            line = line.split()
            if line[2] != 'sil': 
                tokens = [*tokens,' ',line[2]]
        return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]
    except FileNotFoundError:
        raise FileNotFoundError(f"Alignment file not found at path: {path}")

def load_data(video_path: str, alignments_dir: str) -> (tf.Tensor, tf.Tensor):
    try:
        file_name = os.path.basename(video_path).split('.')[0]
        alignment_path = os.path.join(alignments_dir, f"{file_name}.align")
        
        frames = load_video(video_path) 
        alignments = load_alignments(alignment_path)
        
        return frames, alignments
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error loading data: {e}")
