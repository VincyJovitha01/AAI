from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import face_recognition
import os
import sqlite3
from datetime import datetime, date
import json

app = Flask(__name__)
CORS(app)

# Database setup
def init_db():
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    
    # Students table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT UNIQUE,
            name TEXT NOT NULL,
            face_encoding TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Attendance table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT,
            date DATE,
            status TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (student_id) REFERENCES students (student_id)
        )
    ''')
    
    conn.commit()
    conn.close()

class AttendanceSystem:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_ids = []
        self.load_known_faces()
    
    def load_known_faces(self):
        """Load known face encodings from database"""
        conn = sqlite3.connect('attendance.db')
        cursor = conn.cursor()
        cursor.execute("SELECT student_id, name, face_encoding FROM students")
        students = cursor.fetchall()
        conn.close()
        
        self.known_face_encodings = []
        self.known_face_ids = []
        
        for student_id, name, encoding_json in students:
            if encoding_json:
                encoding = json.loads(encoding_json)
                self.known_face_encodings.append(np.array(encoding))
                self.known_face_ids.append(student_id)
    
    def register_student(self, student_id, name, image_file):
        """Register a new student with face encoding"""
        # Load image and find face
        image = face_recognition.load_image_file(image_file)
        face_encodings = face_recognition.face_encodings(image)
        
        if len(face_encodings) == 0:
            return False, "No face detected in the image"
        
        # Store face encoding
        face_encoding = face_encodings[0]
        encoding_json = json.dumps(face_encoding.tolist())
        
        # Save to database
        conn = sqlite3.connect('attendance.db')
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT OR REPLACE INTO students (student_id, name, face_encoding) VALUES (?, ?, ?)",
                (student_id, name, encoding_json)
            )
            conn.commit()
            conn.close()
            
            # Reload known faces
            self.load_known_faces()
            return True, "Student registered successfully"
        except Exception as e:
            conn.close()
            return False, f"Error: {str(e)}"
    
    def mark_attendance(self, image_file, is_live_feed=False):
        """Mark attendance from image or live feed"""
        if is_live_feed:
            # For live feed, image_file is numpy array
            image = image_file
        else:
            image = face_recognition.load_image_file(image_file)
        
        # Find faces in the image
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)
        
        present_students = []
        attendance_results = []
        
        for face_encoding in face_encodings:
            # Compare with known faces
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    student_id = self.known_face_ids[best_match_index]
                    present_students.append(student_id)
                    
                    # Get student name
                    conn = sqlite3.connect('attendance.db')
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM students WHERE student_id = ?", (student_id,))
                    result = cursor.fetchone()
                    student_name = result[0] if result else "Unknown"
                    conn.close()
                    
                    attendance_results.append({
                        'student_id': student_id,
                        'name': student_name,
                        'status': 'Present',
                        'confidence': float(1 - face_distances[best_match_index])
                    })
        
        # Record attendance in database
        today = date.today().isoformat()
        conn = sqlite3.connect('attendance.db')
        cursor = conn.cursor()
        
        # Get all students
        cursor.execute("SELECT student_id FROM students")
        all_students = [row[0] for row in cursor.fetchall()]
        
        for student_id in all_students:
            status = 'Present' if student_id in present_students else 'Absent'
            cursor.execute(
                "INSERT OR REPLACE INTO attendance (student_id, date, status) VALUES (?, ?, ?)",
                (student_id, today, status)
            )
        
        conn.commit()
        conn.close()
        
        return attendance_results

# Initialize system
attendance_system = AttendanceSystem()

@app.route('/api/register', methods=['POST'])
def register_student():
    """Register a new student"""
    if 'image' not in request.files:
        return jsonify({'success': False, 'message': 'No image provided'})
    
    student_id = request.form.get('student_id')
    name = request.form.get('name')
    image_file = request.files['image']
    
    if not student_id or not name:
        return jsonify({'success': False, 'message': 'Student ID and name are required'})
    
    success, message = attendance_system.register_student(student_id, name, image_file)
    return jsonify({'success': success, 'message': message})

@app.route('/api/mark_attendance', methods=['POST'])
def mark_attendance():
    """Mark attendance from uploaded image"""
    if 'image' not in request.files:
        return jsonify({'success': False, 'message': 'No image provided'})
    
    image_file = request.files['image']
    results = attendance_system.mark_attendance(image_file)
    
    return jsonify({
        'success': True,
        'results': results,
        'total_present': len(results),
        'message': f'Attendance marked for {len(results)} students'
    })

@app.route('/api/attendance/<date>', methods=['GET'])
def get_attendance(date):
    """Get attendance for a specific date"""
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT s.student_id, s.name, a.status, a.timestamp 
        FROM students s 
        LEFT JOIN attendance a ON s.student_id = a.student_id AND a.date = ?
        ORDER BY s.student_id
    ''', (date,))
    
    records = cursor.fetchall()
    conn.close()
    
    attendance_data = []
    for record in records:
        attendance_data.append({
            'student_id': record[0],
            'name': record[1],
            'status': record[2] or 'Not Recorded',
            'timestamp': record[3]
        })
    
    return jsonify({'date': date, 'attendance': attendance_data})

@app.route('/api/update_attendance', methods=['POST'])
def update_attendance():
    """Manually update attendance status"""
    data = request.json
    student_id = data.get('student_id')
    date = data.get('date')
    status = data.get('status')
    
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR REPLACE INTO attendance (student_id, date, status) VALUES (?, ?, ?)",
        (student_id, date, status)
    )
    conn.commit()
    conn.close()
    
    return jsonify({'success': True, 'message': 'Attendance updated'})

@app.route('/api/export/<date>', methods=['GET'])
def export_attendance(date):
    """Export attendance as CSV"""
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT s.student_id, s.name, a.status 
        FROM students s 
        LEFT JOIN attendance a ON s.student_id = a.student_id AND a.date = ?
        ORDER BY s.student_id
    ''', (date,))
    
    records = cursor.fetchall()
    conn.close()
    
    csv_data = "Student ID,Name,Status,Date\n"
    for record in records:
        csv_data += f"{record[0]},{record[1]},{record[2] or 'Not Recorded'},{date}\n"
    
    return csv_data, 200, {
        'Content-Type': 'text/csv',
        'Content-Disposition': f'attachment; filename=attendance_{date}.csv'
    }

if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5000)