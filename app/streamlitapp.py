import streamlit as st
import tensorflow as tf
import os
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

        # Construct the paths for the selected video files
        mpg_file_path = os.path.join(data_dir, selected_video)
        mp4_file_path = os.path.join(output_dir, f"{selected_video.split('.')[0]}.mp4")  # Assume MP4 conversion naming convention

        # Render the video and model predictions
        with col1:
            st.info('The video below displays the converted video in MP4 format')
            if os.path.exists(mp4_file_path):
                st.video(mp4_file_path)
            else:
                st.error(f"Video file '{mp4_file_path}' not found. Please convert the selected video to MP4 format.")

        with col2:
            st.info('This is all the machine learning model sees when making a prediction')
            alignment_path = os.path.join(alignments_dir, f"{selected_video.split('.')[0]}.align")

            video_data = load_video(mpg_file_path)
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
