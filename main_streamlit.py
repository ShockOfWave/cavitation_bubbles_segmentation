import streamlit as st
import requests
import os
import subprocess
import tempfile
import uuid
from settings import get_settings


settings = get_settings()

# Используем разные адреса для внутренней связи и для ссылок скачивания
INTERNAL_API_URL = f"http://fastapi:{settings.fastapi_port}"   # для запросов между контейнерами
EXTERNAL_API_URL = f"http://localhost:{settings.fastapi_port}"   # для формирования ссылок, доступных извне

# Инициализируем сессионное состояние для хранения токена
if "token" not in st.session_state:
    st.session_state.token = None

def rerun_or_stop():
    if hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
    else:
        st.stop()

# Форма авторизации
if st.session_state.token is None:
    st.title("Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        if submitted:
            response = requests.post(f"{INTERNAL_API_URL}/token", data={"username": username, "password": password})
            if response.status_code == 200:
                st.session_state.token = response.json()["access_token"]
                st.success("Logged in!")
                rerun_or_stop()
            else:
                st.error("Invalid credentials. Please try again.")
    st.stop()

st.title("Отслеживание кавитационных пузырьков и анализ гистограмм")
uploaded_file = st.file_uploader("Загрузите видео", type=["mp4", "avi", "mov"])
if uploaded_file is not None:
    video_bytes = uploaded_file.getvalue()
    ext = uploaded_file.name.split('.')[-1].lower()

    # Предпросмотр видео: если формат не mp4, пробуем перекодировать для предпросмотра
    if ext != "mp4":
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_preview_path = tmp.name
        original_path = os.path.join("uploads", f"{uuid.uuid4()}_{uploaded_file.name}")
        os.makedirs("uploads", exist_ok=True)
        with open(original_path, "wb") as f:
            f.write(video_bytes)
        try:
            subprocess.run(
                ["ffmpeg", "-i", original_path, "-c:v", "libx264", "-preset", "fast", "-y", tmp_preview_path],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            with open(tmp_preview_path, "rb") as f:
                preview_bytes = f.read()
            st.video(preview_bytes)
        except Exception as e:
            st.error(f"Ошибка преобразования видео: {e}")
    else:
        st.video(video_bytes)

    # Сохраняем оригинал для отправки на обработку
    unique_filename = f"{uuid.uuid4()}_{uploaded_file.name}"
    save_path = os.path.join("uploads", unique_filename)
    os.makedirs("uploads", exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(video_bytes)

    if st.button("Обработать видео"):
        headers = {"Authorization": f"Bearer {st.session_state.token}"}
        files = {"file": (unique_filename, video_bytes)}
        with st.spinner("Обработка видео..."):
            response = requests.post(f"{INTERNAL_API_URL}/process_video/", files=files, headers=headers)
        if response.status_code == 200:
            result = response.json()
            output_video = result.get("output_video")
            csv_file = result.get("csv_file")
            speed_hist = result.get("speed_hist_file")
            area_hist = result.get("area_hist_file")
            st.success("Видео обработано!")

            # Если обработанное видео в формате AVI, перекодируем в mp4 для предпросмотра
            if output_video.lower().endswith(".avi"):
                mp4_video = output_video.replace(".avi", ".mp4")
                try:
                    subprocess.run(
                        ["ffmpeg", "-i", output_video, "-c:v", "libx264", "-preset", "fast", "-y", mp4_video],
                        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                    )
                    output_video = mp4_video
                except Exception as e:
                    st.error(f"Ошибка перекодировки видео: {e}")
            st.video(output_video)
            # Формируем ссылки для скачивания, используя EXTERNAL_API_URL
            st.markdown(f"[Скачать обработанное видео]({EXTERNAL_API_URL}/download/?path={output_video}&token={st.session_state.token})")
            st.markdown(f"[Скачать CSV файл]({EXTERNAL_API_URL}/download/?path={csv_file}&token={st.session_state.token})")
            st.markdown(f"[Скачать гистограмму скорости]({EXTERNAL_API_URL}/download/?path={speed_hist}&token={st.session_state.token})")
            st.markdown(f"[Скачать гистограмму площади]({EXTERNAL_API_URL}/download/?path={area_hist}&token={st.session_state.token})")
        else:
            st.error("Ошибка при обработке видео.")
