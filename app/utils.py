import tensorflow as tf
import cv2
import os 

# Example of vocabulary and lookup layers
vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

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
