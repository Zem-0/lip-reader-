import streamlit as st
import tensorflow as tf
import os
import imageio
from utils import load_data, num_to_char
from modelutil import load_model


st.set_page_config(layout='wide')
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 's1')
alignments_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'alignments', 's1')
options = os.listdir(data_dir)

# Setup the sidebar
with st.sidebar:
    st.image('logoipsum-247.svg')
    st.title('Lip Reader')
    st.info('This application is originally developed from the LipNet deep learning model.')
    selected_video = st.selectbox('Choose video', options)


# Main content
col1, col2 = st.columns([2,3])
with col1:
    st.image('logoipsum-247.svg',width=100)
    st.title('LipReader App')



with col2:
    st.info('This application is originally developed from the LipNet deep learning model.Lipreading is the task of decoding text from the movement of a speakers mouth. Traditional approaches separated the problem into two stages: designing or learning visual features, and prediction.LipNet achieves 95.2% accuracy in sentence-level, overlapped speaker split task, outperforming experienced human lipreaders and the previous 86.4% word-level state-of-the-art accuracy ',icon="ℹ️")

# Check if the data directory exists
if not os.path.exists(data_dir):
    st.error(f"The data directory {data_dir} does not exist.")
else:
    # Generating a list of options or videos 
    options = os.listdir(data_dir)
    
    if options:
        

        # Generate two columns 
        col1, col2 = st.columns(2)

        file_path = os.path.join(data_dir, selected_video)
        alignment_path = os.path.join(alignments_dir, f"{selected_video.split('.')[0]}.align")

        # Rendering the video 
        with col1:
            st.info('The video below displays the converted video in mp4 format')
            os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')

            # Rendering inside of the app
            with open('test_video.mp4', 'rb') as video:
                video_bytes = video.read()
            st.video(video_bytes)

            with col2:
                st.info('This is all the machine learning model sees when making a prediction')
                try:
                    frames, alignments = load_data(file_path, alignments_dir)
                    #imageio.mimsave('animation.gif', frames, fps=10)
                    st.image('app\\animation.gif', width=400)

                    st.info('This is the output of the machine learning model as tokens')
                    model = load_model()
                    yhat = model.predict(tf.expand_dims(frames, axis=0))
                    decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
                    st.text(decoder)

                    # Convert prediction to text
                    st.info('Decode the raw tokens into words')
                    converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
                    st.text(converted_prediction)
                except FileNotFoundError as e:
                    st.error(f"FileNotFoundError: {e}")

    else:
        st.warning("No video files found in the data directory.")