"""Database models for the Sign Language Recognition application."""
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()


class User(db.Model):
    """User accounts."""
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=True)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    sessions = db.relationship('RecognitionSession', backref='user', lazy=True)
    recorded_gestures = db.relationship('RecordedGesture', backref='user', lazy=True)
    learning_progress = db.relationship('LearningProgress', backref='user', lazy=True)


class RecognitionSession(db.Model):
    """Recognition sessions."""
    __tablename__ = 'sessions'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    session_token = db.Column(db.String(100), nullable=False)
    language = db.Column(db.String(10), default='ASL')
    start_time = db.Column(db.DateTime, default=datetime.utcnow)
    end_time = db.Column(db.DateTime, nullable=True)
    total_signs = db.Column(db.Integer, default=0)
    avg_confidence = db.Column(db.Float, default=0.0)
    gesture_logs = db.relationship('GestureLog', backref='session', lazy=True)
    sentences = db.relationship('Sentence', backref='session', lazy=True)


class GestureLog(db.Model):
    """Individual gesture detections."""
    __tablename__ = 'gesture_logs'
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('sessions.id'), nullable=False)
    person_id = db.Column(db.Integer, default=0)
    gesture = db.Column(db.String(20), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    emotion = db.Column(db.String(20), nullable=True)
    landmarks_data = db.Column(db.Text, nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)


class Sentence(db.Model):
    """Formed sentences."""
    __tablename__ = 'sentences'
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('sessions.id'), nullable=False)
    raw_text = db.Column(db.Text, nullable=False)
    corrected_text = db.Column(db.Text, nullable=True)
    spoken = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class RecordedGesture(db.Model):
    """User-recorded custom gestures."""
    __tablename__ = 'recorded_gestures'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    gesture_name = db.Column(db.String(50), nullable=False)
    landmarks_json = db.Column(db.Text, nullable=False)
    image_path = db.Column(db.String(255), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class LearningProgress(db.Model):
    """Teaching mode progress tracking."""
    __tablename__ = 'learning_progress'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    language = db.Column(db.String(10), default='ASL')
    gesture = db.Column(db.String(20), nullable=False)
    attempts = db.Column(db.Integer, default=0)
    correct = db.Column(db.Integer, default=0)
    best_confidence = db.Column(db.Float, default=0.0)
    last_practiced = db.Column(db.DateTime, default=datetime.utcnow)


def init_db(app):
    """Initialize the database."""
    db.init_app(app)
    with app.app_context():
        db.create_all()
