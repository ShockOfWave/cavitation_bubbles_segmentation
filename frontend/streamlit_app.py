import base64
import os
import subprocess
import tempfile

import requests
import streamlit as st

from src.config import get_settings

settings = get_settings()

INTERNAL_API_URL = f"http://{settings.fastapi_host}:{settings.fastapi_port}"
REQUEST_TIMEOUT = 600  # seconds

if "token" not in st.session_state:
    st.session_state.token = None
if "processing_result" not in st.session_state:
    st.session_state.processing_result = None


if st.session_state.token is None:
    st.title("Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        if submitted:
            response = requests.post(
                f"{INTERNAL_API_URL}/token",
                data={"username": username, "password": password},
                timeout=30,
            )
            if response.status_code == 200:
                st.session_state.token = response.json()["access_token"]
                st.success("Logged in!")
                st.rerun()
            else:
                st.error("Invalid credentials. Please try again.")
    st.stop()

st.title("Отслеживание кавитационных пузырьков и анализ гистограмм")
uploaded_file = st.file_uploader("Загрузите видео", type=["mp4", "avi", "mov"])
if uploaded_file is not None:
    video_bytes = uploaded_file.getvalue()
    ext = uploaded_file.name.rsplit(".", 1)[-1].lower()

    if ext != "mp4":
        tmp_orig_path = None
        tmp_preview_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp_orig:
                tmp_orig.write(video_bytes)
                tmp_orig_path = tmp_orig.name
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_preview:
                tmp_preview_path = tmp_preview.name
            subprocess.run(
                ["ffmpeg", "-i", tmp_orig_path, "-c:v", "libx264", "-preset", "fast", "-y", tmp_preview_path],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            with open(tmp_preview_path, "rb") as f:
                preview_bytes = f.read()
            st.video(preview_bytes)
        except Exception as e:
            st.error(f"Ошибка преобразования видео: {e}")
        finally:
            if tmp_orig_path and os.path.exists(tmp_orig_path):
                os.unlink(tmp_orig_path)
            if tmp_preview_path and os.path.exists(tmp_preview_path):
                os.unlink(tmp_preview_path)
    else:
        st.video(video_bytes)

    if st.button("Обработать видео"):
        headers = {"Authorization": f"Bearer {st.session_state.token}"}
        files = {"file": (uploaded_file.name, video_bytes)}
        with st.spinner("Обработка видео..."):
            response = requests.post(
                f"{INTERNAL_API_URL}/process_video/",
                files=files,
                headers=headers,
                timeout=REQUEST_TIMEOUT,
            )
        if response.status_code == 200:
            result = response.json()
            video_data = base64.b64decode(result["output_video"])
            csv_data = base64.b64decode(result["csv_file"])

            downloads = {
                "Скачать обработанное видео": {"data": video_data, "file_name": result["output_video_name"]},
                "Скачать CSV файл": {"data": csv_data, "file_name": result["csv_file_name"]},
            }
            if result.get("speed_hist_file"):
                downloads["Скачать гистограмму скорости"] = {
                    "data": base64.b64decode(result["speed_hist_file"]),
                    "file_name": result["speed_hist_name"],
                }
            if result.get("area_hist_file"):
                downloads["Скачать гистограмму площади"] = {
                    "data": base64.b64decode(result["area_hist_file"]),
                    "file_name": result["area_hist_name"],
                }
            st.session_state.processing_result = {
                "video_data": video_data,
                "downloads": downloads,
            }
        else:
            st.error("Ошибка при обработке видео.")

    if st.session_state.processing_result is not None:
        result = st.session_state.processing_result
        st.success("Видео обработано!")
        st.video(result["video_data"])
        for label, info in result["downloads"].items():
            st.download_button(
                label=label,
                data=info["data"],
                file_name=info["file_name"],
                key=label,
            )
