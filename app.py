import streamlit as st
from contador_carros import contar_carros
import os

# Caminho direto do vídeo salvo localmente
VIDEO_PATH = "video/5927708-hd_1080_1920_30fps.mp4"

st.set_page_config(page_title="🚗 Contador de Carros", layout="centered")
st.title("🚗 Contador de Carros em Vídeo (Streamlit + YOLOv8)")

if not os.path.exists(VIDEO_PATH):
    st.error("O vídeo não foi encontrado. Verifique se está em 'video/5927708-hd_1080_1920_30fps.mp4'")
else:
    stframe = st.empty()
    with st.spinner("Processando vídeo..."):
        for frame in contar_carros(VIDEO_PATH):
            stframe.image(frame, channels="BGR")
    st.success("Vídeo finalizado.")
