import streamlit as st
from contador_carros import contar_carros
import os

#------------ Caminho do vídeo de entrada (deve estar dentro da pasta 'video/')
CAMINHO_VIDEO = "video/5927708-hd_1080_1920_30fps.mp4"

st.set_page_config(page_title="🚗 Contador de Carros", layout="centered")
st.title("🚗 Contador de Carros em Vídeo (Streamlit + YOLOv8)")

#------------ Verifica se o vídeo existe no caminho esperado
if not os.path.exists(CAMINHO_VIDEO):
    st.error("O vídeo não foi encontrado. Verifique se está em 'video/5927708-hd_1080_1920_30fps.mp4'")
else:
    quadro_stream = st.empty()

    with st.spinner("Processando vídeo, aguarde..."):
        for imagem in contar_carros(CAMINHO_VIDEO):
            quadro_stream.image(imagem, channels="BGR")

    st.success("Vídeo finalizado.")
