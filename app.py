"""Main Flask + SocketIO application for Sign Language Recognition."""
import os
import sys
import json
import time
import base64
import uuid
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify, Response
from flask_socketio import SocketIO, emit
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from database import db, init_db, RecognitionSession, GestureLog, Sentence, LearningProgress
from models.hand_detector import HandDetector
from models.sign_classifier import SignClassifier
from models.emotion_detector import EmotionDetector
from models.multi_person_tracker import MultiPersonTracker, PersonState
from models.sentence_former import SentenceFormer
from utils.helpers import generate_session_token, FPSCounter, format_fps, ensure_dirs

app = Flask(__name__)
app.config.from_object(Config)

init_db(app)
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='eventlet')

ensure_dirs(
    Config.MODELS_DIR, Config.DATASETS_DIR,
    Config.UPLOAD_FOLDER, Config.RECORDING_FOLDER,
    os.path.join(app.static_folder, 'img'),
    os.path.join(app.static_folder, 'fonts')
)

hand_detector = HandDetector(
    yolo_model_path=Config.YOLO_MODEL_PATH if os.path.exists(Config.YOLO_MODEL_PATH) else None,
    max_hands=2, detection_confidence=0.7
)
sign_classifier = SignClassifier(
    model_path=Config.SIGN_MODEL_PATH if os.path.exists(Config.SIGN_MODEL_PATH) else None,
    num_classes=len(Config.ASL_CLASSES)
)
emotion_detector = EmotionDetector(
    model_path=Config.EMOTION_MODEL_PATH if os.path.exists(Config.EMOTION_MODEL_PATH) else None
)
multi_tracker = MultiPersonTracker(max_persons=Config.MAX_PERSONS)
sentence_former = SentenceFormer()
fps_counter = FPSCounter()
active_sessions = {}


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recognize')
def recognize():
    return render_template('recognize.html')

@app.route('/learn')
def learn():
    return render_template('learn.html')

@app.route('/history')
def history():
    sessions = RecognitionSession.query.order_by(RecognitionSession.start_time.desc()).limit(50).all()
    return render_template('history.html', sessions=sessions)

@app.route('/dashboard')
def dashboard():
    total_sessions = RecognitionSession.query.count()
    total_gestures = GestureLog.query.count()
    total_sentences = Sentence.query.count()
    recent_sessions = RecognitionSession.query.order_by(RecognitionSession.start_time.desc()).limit(5).all()
    return render_template('dashboard.html', total_sessions=total_sessions, total_gestures=total_gestures, total_sentences=total_sentences, recent_sessions=recent_sessions)

@app.route('/settings')
def settings():
    return render_template('settings.html')


@app.route('/api/session/start', methods=['POST'])
def start_session():
    data = request.json or {}
    language = data.get('language', 'ASL')
    session = RecognitionSession(session_token=generate_session_token(), language=language, start_time=datetime.utcnow())
    db.session.add(session)
    db.session.commit()
    active_sessions[session.session_token] = {'id': session.id, 'language': language, 'start_time': time.time()}
    return jsonify({'session_token': session.session_token, 'id': session.id})

@app.route('/api/session/end', methods=['POST'])
def end_session():
    data = request.json or {}
    token = data.get('session_token')
    if token in active_sessions:
        session = RecognitionSession.query.get(active_sessions[token]['id'])
        if session:
            session.end_time = datetime.utcnow()
            db.session.commit()
        del active_sessions[token]
    return jsonify({'status': 'ok'})

@app.route('/api/history', methods=['GET'])
def get_history():
    sessions = RecognitionSession.query.order_by(RecognitionSession.start_time.desc()).limit(50).all()
    result = []
    for s in sessions:
        gestures = GestureLog.query.filter_by(session_id=s.id).count()
        sentences = Sentence.query.filter_by(session_id=s.id).all()
        result.append({'id': s.id, 'language': s.language, 'start_time': s.start_time.isoformat() if s.start_time else '', 'end_time': s.end_time.isoformat() if s.end_time else '', 'total_gestures': gestures, 'avg_confidence': round(s.avg_confidence, 2), 'sentences': [{'raw': sent.raw_text, 'corrected': sent.corrected_text} for sent in sentences]})
    return jsonify(result)

@app.route('/api/stats', methods=['GET'])
def get_stats():
    total_sessions = RecognitionSession.query.count()
    total_gestures = GestureLog.query.count()
    total_sentences = Sentence.query.count()
    avg_confidence = 0
    if total_gestures > 0:
        result = db.session.query(db.func.avg(GestureLog.confidence)).scalar()
        avg_confidence = round(float(result or 0), 2)
    return jsonify({'total_sessions': total_sessions, 'total_gestures': total_gestures, 'total_sentences': total_sentences, 'avg_confidence': avg_confidence})

@app.route('/api/classes/<language>')
def get_classes(language):
    classes_map = {'ASL': Config.ASL_CLASSES, 'ISL': Config.ISL_CLASSES, 'ArSL': Config.ARSL_CLASSES}
    return jsonify(classes_map.get(language.upper(), Config.ASL_CLASSES))


@socketio.on('connect')
def handle_connect():
    print(f'[WebSocket] Client connected: {request.sid}')
    emit('connection_status', {'status': 'connected', 'sid': request.sid})

@socketio.on('disconnect')
def handle_disconnect():
    print(f'[WebSocket] Client disconnected: {request.sid}')

@socketio.on('process_frame')
def handle_frame(data):
    try:
        fps_counter.tick()
        img_data = data.get('image', '')
        if ',' in img_data: img_data = img_data.split(',')[1]
        img_bytes = base64.b64decode(img_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            emit('prediction_result', {'error': 'Invalid frame'})
            return
        session_token = data.get('session_token', '')
        persons = hand_detector.detect_persons(frame)
        multi_tracker.update(persons)
        hand_results = hand_detector.detect_hands(frame)
        face_results = hand_detector.detect_faces(frame)
        person_predictions = []
        for tid, person_state in multi_tracker.get_all_persons().items():
            prediction = {'track_id': tid, 'bbox': person_state.last_bbox, 'gesture': 'nothing', 'confidence': 0.0, 'emotion': 'Neutral', 'emoji': '\U0001f610', 'current_text': person_state.get_current_text(), 'char_buffer': ''.join(person_state.char_buffer)}
            if hand_results:
                hand = hand_results[0]
                landmarks = hand['normalized_landmarks']
                flat_landmarks = np.array(landmarks).flatten()
                sign, confidence, probs = sign_classifier.predict(image=hand.get('hand_crop'), landmarks=flat_landmarks)
                prediction['gesture'] = sign
                prediction['confidence'] = confidence
                new_char = person_state.add_prediction(sign, confidence, stability_window=Config.STABILITY_WINDOW, confidence_threshold=Config.CONFIDENCE_THRESHOLD)
                prediction['current_text'] = person_state.get_current_text()
                prediction['char_buffer'] = ''.join(person_state.char_buffer)
                prediction['new_char'] = new_char
            if face_results:
                face = face_results[0]
                emotion, em_conf, emoji, scores = emotion_detector.predict(face['face_crop_gray'])
                prediction['emotion'] = emotion
                prediction['emoji'] = emoji
                prediction['emotion_confidence'] = em_conf
                person_state.emotion = emotion
                person_state.emoji = emoji
            person_predictions.append(prediction)
            if session_token in active_sessions and prediction['gesture'] != 'nothing':
                try:
                    log = GestureLog(session_id=active_sessions[session_token]['id'], person_id=tid, gesture=prediction['gesture'], confidence=prediction['confidence'], emotion=prediction['emotion'])
                    db.session.add(log)
                    db.session.commit()
                except Exception:
                    db.session.rollback()
        current_fps = fps_counter.get_fps()
        response = {'persons': person_predictions, 'fps': format_fps(current_fps), 'total_persons': len(person_predictions), 'hand_detected': len(hand_results) > 0, 'face_detected': len(face_results) > 0}
        emit('prediction_result', response)
    except Exception as e:
        print(f'[Error] Frame processing failed: {e}')
        import traceback
        traceback.print_exc()
        emit('prediction_result', {'error': str(e)})

@socketio.on('process_landmarks')
def handle_landmarks(data):
    try:
        fps_counter.tick()
        landmarks = data.get('landmarks', [])
        session_token = data.get('session_token', '')
        face_data = data.get('face', None)
        if not landmarks:
            emit('prediction_result', {'error': 'No landmarks'})
            return
        lm_array = np.array(landmarks).flatten()
        sign, confidence, probs = sign_classifier.predict(landmarks=lm_array)
        if 0 not in multi_tracker.persons:
            multi_tracker.persons[0] = PersonState(0)
        person = multi_tracker.persons[0]
        new_char = person.add_prediction(sign, confidence, stability_window=Config.STABILITY_WINDOW, confidence_threshold=Config.CONFIDENCE_THRESHOLD)
        emotion = 'Neutral'
        emoji = '\U0001f610'
        if face_data:
            face_arr = np.array(face_data, dtype=np.uint8).reshape(48, 48)
            emotion, _, emoji, _ = emotion_detector.predict(face_arr)
        current_fps = fps_counter.get_fps()
        response = {'persons': [{'track_id': 0, 'gesture': sign, 'confidence': confidence, 'emotion': emotion, 'emoji': emoji, 'current_text': person.get_current_text(), 'char_buffer': ''.join(person.char_buffer), 'new_char': new_char}], 'fps': format_fps(current_fps), 'total_persons': 1, 'hand_detected': True, 'face_detected': face_data is not None}
        emit('prediction_result', response)
    except Exception as e:
        print(f'[Error] Landmark processing failed: {e}')
        emit('prediction_result', {'error': str(e)})

@socketio.on('save_sentence')
def handle_save_sentence(data):
    try:
        raw_text = data.get('raw_text', '')
        session_token = data.get('session_token', '')
        corrected = sentence_former.form_sentence(raw_text.split())
        if session_token in active_sessions:
            sentence = Sentence(session_id=active_sessions[session_token]['id'], raw_text=raw_text, corrected_text=corrected)
            db.session.add(sentence)
            db.session.commit()
        emit('sentence_saved', {'raw': raw_text, 'corrected': corrected, 'suggestions': sentence_former.get_suggestions(raw_text.split()[-1] if raw_text.split() else '')})
    except Exception as e:
        emit('sentence_saved', {'error': str(e)})

@socketio.on('clear_text')
def handle_clear_text(data):
    track_id = data.get('track_id', None)
    if track_id is not None:
        person = multi_tracker.get_person(track_id)
        if person: person.clear()
    else:
        for person in multi_tracker.get_all_persons().values():
            person.clear()
    emit('text_cleared', {'status': 'ok'})


if __name__ == '__main__':
    print('\n' + '='*60)
    print('  \U0001f91f Sign Language Recognition System')
    print('  Emotion-Aware \u2022 Multi-Person \u2022 Real-Time')
    print('='*60)
    print(f'  Server: http://localhost:5000')
    print('='*60 + '\n')
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)
