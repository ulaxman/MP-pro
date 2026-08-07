import os

class Config:
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    SECRET_KEY = os.environ.get('SECRET_KEY', 'sign-language-secret-key-2024')
    SQLALCHEMY_DATABASE_URI = f'sqlite:///{os.path.join(BASE_DIR, "database.db")}'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')
    SIGN_MODEL_PATH = os.path.join(MODELS_DIR, 'sign_classifier.h5')
    EMOTION_MODEL_PATH = os.path.join(MODELS_DIR, 'emotion_model.h5')
    YOLO_MODEL_PATH = os.path.join(MODELS_DIR, 'yolov8n.pt')
    DATASETS_DIR = os.path.join(BASE_DIR, 'datasets')
    ASL_DATASET = os.path.join(DATASETS_DIR, 'asl_alphabet')
    MNIST_DATASET = os.path.join(DATASETS_DIR, 'sign_mnist')
    ISL_DATASET = os.path.join(DATASETS_DIR, 'indian_sl')
    ARSL_DATASET = os.path.join(DATASETS_DIR, 'arabic_sl')
    FER_DATASET = os.path.join(DATASETS_DIR, 'fer2013')
    CAMERA_WIDTH = 1920
    CAMERA_HEIGHT = 1080
    CAMERA_FPS = 30
    CONFIDENCE_THRESHOLD = 0.75
    STABILITY_WINDOW = 5
    MAX_PERSONS = 4
    ASL_CLASSES = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') + ['space', 'del', 'nothing']
    ISL_CLASSES = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') + list('0123456789')
    ARSL_CLASSES = ['ain','al','aleff','bb','dal','dha','dhad','fa','gaaf','ghain','ha','haa','jeem','kaaf','khaa','la','laam','meem','nun','ra','saad','seen','sheen','ta','taa','thaa','thal','toot','waw','ya','yaa','zay']
    EMOTION_CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    EMOTION_EMOJIS = {'Angry': '😠', 'Disgust': '🤢', 'Fear': '😨', 'Happy': '😊', 'Sad': '😢', 'Surprise': '😲', 'Neutral': '😐'}
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    RECORDING_FOLDER = os.path.join(BASE_DIR, 'recordings')
