import numpy as np
import streamlit as st
import tensorflow as tf
import os
import imageio
from utils import load_video, load_alignments, num_to_char
from modelutil import load_model

# Custom CSS to reduce gap between columns
st.markdown("""
    <style>
    .css-145kmo2 .st-df {
        display: flex;
    }
    .css-145kmo2 .st-df .st-cb {
        flex: 0 0 auto;
        margin-right: 1em;
    }
    </style>
    """, unsafe_allow_html=True)

# Setup the sidebar
with st.sidebar:
    st.image('logoipsum-247.svg', width=200)
    st.title('Lip Reader')
    st.info('This application is originally developed from the LipNet deep learning model.')

# Main content
col1, col2 = st.columns(2)
with col1:
    st.image('logoipsum-247.svg')
    st.title('Lip Reader')

with col2:
    st.title('LipNet Full Stack App')
    st.info('This application is originally developed from the LipNet deep learning model.')

# Define the data directory paths
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 's1'))
alignments_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'alignments', 's1'))

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

        file_path = os.path.join(data_dir, selected_video)
        alignment_path = os.path.join(alignments_dir, f"{selected_video.split('.')[0]}.align")

        # Render the video and model predictions
        with col1:
            st.info('The video below displays the converted video in mp4 format')
            video_data = load_video(file_path)

            st.info('The video below displays the converted video in mp4 format')
            os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')

            # Rendering inside of the app
            with open('test_video.mp4', 'rb') as video:
                video_bytes = video.read()
            st.video(video_bytes)


        with col2:
            st.info('This is all the machine learning model sees when making a prediction')
            alignments_data = load_alignments(alignment_path)
            
            model = load_model()
            yhat = model.predict(tf.expand_dims(video_data, axis=0))
            decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
            st.text(decoder)

            # Convert prediction to text
            st.info('Decode the raw tokens into words')
            converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
            st.text(converted_prediction)

    else:
        st.warning("No video files found in the data directory.")
