from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from datetime import datetime, timedelta
import base64
import shutil
import os
import uuid
import tempfile

from src.video_processing import VideoProcessor
from settings import get_settings, Settings

# Получаем настройки (включая username и password из .env)
settings = get_settings()

# Константы для JWT
SECRET_KEY = settings.secret_key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

app = FastAPI()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
         expire = datetime.utcnow() + expires_delta
    else:
         expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_password(plain_password: str, stored_password: str) -> bool:
    # Здесь простое сравнение строк, в продакшене рекомендуется использовать хэширование
    return plain_password == stored_password

def authenticate_user(username: str, password: str):
    if username == settings.username and verify_password(password, settings.password):
         return {"username": username}
    return None

@app.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
         raise HTTPException(
             status_code=status.HTTP_401_UNAUTHORIZED,
             detail="Incorrect username or password",
             headers={"WWW-Authenticate": "Bearer"},
         )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
         data={"sub": user["username"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
         status_code=status.HTTP_401_UNAUTHORIZED,
         detail="Could not validate credentials",
         headers={"WWW-Authenticate": "Bearer"},
    )
    try:
         payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
         username: str = payload.get("sub")
         if username is None:
             raise credentials_exception
    except JWTError:
         raise credentials_exception
    return {"username": username}

# Инициализируем VideoProcessor (укажите корректный путь к модели)
MODEL_PATH = "hf_model_repo/model.pt"
video_processor = VideoProcessor(MODEL_PATH)

def _read_file_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

@app.post("/process_video/")
async def process_video_endpoint(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    tmp_dir = tempfile.mkdtemp()
    try:
        unique_filename = f"{uuid.uuid4()}_{file.filename}"
        input_path = os.path.join(tmp_dir, unique_filename)
        with open(input_path, "wb") as buffer:
             shutil.copyfileobj(file.file, buffer)

        output_video_path = os.path.join(tmp_dir, f"processed_{unique_filename}")
        csv_path = os.path.join(tmp_dir, f"data_{unique_filename.split('.')[0]}.csv")

        speed_hist_file, area_hist_file = video_processor.process_video(
             input_path, output_video_path, csv_path, tmp_dir
        )

        return {
             "output_video": _read_file_b64(output_video_path),
             "output_video_name": os.path.basename(output_video_path),
             "csv_file": _read_file_b64(csv_path),
             "csv_file_name": os.path.basename(csv_path),
             "speed_hist_file": _read_file_b64(speed_hist_file) if speed_hist_file else None,
             "speed_hist_name": os.path.basename(speed_hist_file) if speed_hist_file else None,
             "area_hist_file": _read_file_b64(area_hist_file) if area_hist_file else None,
             "area_hist_name": os.path.basename(area_hist_file) if area_hist_file else None,
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
