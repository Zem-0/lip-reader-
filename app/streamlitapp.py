import streamlit as st
import tensorflow as tf
import os
from utils import load_video, load_alignments, num_to_char
from modelutil import load_model

# Function to fetch align text excluding first and last word
def fetch_align_text(selected_video, alignments_dir):
    align_file_path = os.path.join(alignments_dir, f"{selected_video.split('.')[0]}.align")
    align_text = ""

    if os.path.exists(align_file_path):
        with open(align_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines[1:-1]:  # Exclude first and last line
                parts = line.split()
                if len(parts) >= 3:
                    word = parts[2]
                    align_text += f"{word} "
    else:
        st.error(f"Alignment file '{align_file_path}' not found.")

    return align_text.strip()


# Setup the sidebar
with st.sidebar:
    st.image('logoipsum-247.svg')
    st.title('Lip Reader')
    st.info('This application is originally developed from the LipNet deep learning model.')

# Main content
col1, col2 = st.columns(2)
with col1:
    st.image('logoipsum-247.svg',width=100)
    st.title('Lip Reader Application')

with col2:
    st.info('Lipreading is the task of decoding text from the movement of a speakers mouth. Traditional approaches separated the problem into two stages: designing or learning visual features, and prediction. LipNet achieves 95.2% accuracy in sentence-level, overlapped speaker split task, outperforming experienced human lipreaders and the previous 86.4% word-level state-of-the-art accuracy')

# Define the data directory paths
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 's1'))
alignments_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'alignments', 's1'))
output_dir = 'output'  # Directory where converted MP4 videos are stored

# Check if the data directory exists
if not os.path.exists(data_dir):
    st.error(f"The data directory {data_dir} does not exist.")
else:
    # Generating a list of options or videos 
    options = os.listdir(data_dir)
    
    if options:
        # Use session state to track selected video
        if 'selected_video' not in st.session_state:
            st.session_state.selected_video = options[0]

        selected_video = st.selectbox('Choose video', options, index=options.index(st.session_state.selected_video))

        # Clear cache if video selection changes
        if selected_video != st.session_state.selected_video:
            st.session_state.selected_video = selected_video
            st.experimental_rerun()

        # Generate two columns 
        col1, col2 = st.columns(2)
        mpg_file_path = os.path.join(data_dir, selected_video)
        mp4_file_path = os.path.join(output_dir, f"{selected_video.split('.')[0]}.mp4")
        alignment_path = os.path.join(alignments_dir, f"{selected_video.split('.')[0]}.align")

        # Fetch align text excluding first and last word
        align_text = fetch_align_text(selected_video, alignments_dir)

        # Render the video and model predictions
        with col1:
            st.info('The video below displays the converted video in MP4 format')
            if os.path.exists(mp4_file_path):
                st.video(mp4_file_path)
            else:
                st.error(f"Video file '{mp4_file_path}' not found. Please convert the selected video to MP4 format.")

        with col2:
            st.info('This is all the machine learning model sees when making a prediction')
            st.image('app/animation.gif', width=400)
            st.info('This is the array form of the above gif data use by the model')

            video_data = load_video(mpg_file_path)
            
            model = load_model()
            yhat = model.predict(tf.expand_dims(video_data, axis=0))
            decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
            st.text(decoder)


        col3,col4=st.columns(2)
        with col3:
            # Convert prediction to text
            st.info('After decoding the tokens we get the predicted text')
            converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
            st.text(converted_prediction)
        with col4:
            st.info('Original Text')
            st.text(align_text)


    else:
        st.warning("No video files found in the data directory.")
