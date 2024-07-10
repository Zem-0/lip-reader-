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

# Define the data directory paths
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 's1'))
alignments_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'alignments', 's1'))
output_dir = 'output'  # Directory where converted MP4 videos are stored

# Setup the sidebar
with st.sidebar:
    st.image('logoipsum-247.svg')
    st.title('Lip Reader')
    st.info('This application is originally developed from the LipNet deep learning model.')

    # Option to upload a video
    uploaded_file = st.file_uploader("Upload your own video", type=["mp4", "mpg"])

    if uploaded_file is not None:
        # Save the uploaded file temporarily
        temp_file_path = os.path.join(output_dir, uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        selected_video = uploaded_file.name
        mpg_file_path = temp_file_path
        mp4_file_path = temp_file_path
    else:
        options = os.listdir(data_dir)
        if options:
            if 'selected_video' not in st.session_state:
                st.session_state.selected_video = options[0]

            selected_video = st.selectbox('Choose video', options, index=options.index(st.session_state.selected_video))

            if selected_video != st.session_state.selected_video:
                st.session_state.selected_video = selected_video
                st.experimental_rerun()

            mpg_file_path = os.path.join(data_dir, selected_video)
            mp4_file_path = os.path.join(output_dir, f"{selected_video.split('.')[0]}.mp4")
        else:
            st.warning("No video files found in the data directory.")
            selected_video, mpg_file_path, mp4_file_path = None, None, None

# Main content
if selected_video and mpg_file_path and mp4_file_path:
    col1, col2 = st.columns(2)
    with col1:
        st.image('logoipsum-247.svg', width=100)
        st.title('Lip Reader Application')

    with col2:
        st.info('Lipreading is the task of decoding text from the movement of a speakers mouth. Traditional approaches separated the problem into two stages: designing or learning visual features, and prediction. LipNet achieves 95.2% accuracy in sentence-level, overlapped speaker split task, outperforming experienced human lipreaders and the previous 86.4% word-level state-of-the-art accuracy')

    # Fetch align text excluding first and last word
    if uploaded_file is None:
        alignment_path = os.path.join(alignments_dir, f"{selected_video.split('.')[0]}.align")
        align_text = fetch_align_text(selected_video, alignments_dir)
    else:
        align_text = "No alignment text available for uploaded videos."

    # Render the video and model predictions
    col1, col2 = st.columns(2)
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

    col3, col4 = st.columns(2)
    with col3:
        st.info('After decoding the tokens we get the predicted text')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)
    with col4:
        st.info('Original Text')
        st.text(align_text)
