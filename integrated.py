import os
import threading
import queue
import cv2
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, render_template, Response, request, redirect, url_for, session, flash, jsonify, send_from_directory
from email.message import EmailMessage
from PIL import Image
import smtplib
from flask_socketio import SocketIO, emit
import sys
import bcrypt
import imghdr
from fpdf import FPDF
import glob

CREDENTIALS_FILE = 'credentials.txt'

# Email and plain text password
email = 'admin@gmail.com'
password = 'password123'

# Hash the password
hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

# Store the email and hashed password
credentials = f"{email},{hashed_password}"

print(credentials)

app = Flask(__name__)
socketio = SocketIO(app)
app.secret_key = 'your_secret_key'  # Set your secret key for session management

# Email configuration
Sender_Email = "group.theta2024@gmail.com"
Password = "knvq hccb kqts wlrt"  # Use app-specific password if using Gmail

image_queue = queue.Queue()
new_image_event = threading.Event()

last_print_times = {"motion": datetime.min, "face": datetime.min, "unknown_face": datetime.min}
print_interval = timedelta(seconds=10)
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file_path = None
log_file = None

def restart_flask_app():
    python = sys.executable
    os.execl(python, python, *sys.argv)

def get_log_file_path():
    current_date = datetime.now().strftime('%B %d, %Y')  
    return os.path.join(log_dir, f'{current_date}.txt')

def log_message(message):
    global log_file, log_file_path
    current_date = datetime.now().strftime('%B %d, %Y') 
    new_log_file_path = get_log_file_path()
    if new_log_file_path != log_file_path:
        if log_file:
            log_file.close()
        log_file_path = new_log_file_path
        log_file = open(log_file_path, 'a')
    current_time = datetime.now().strftime('%H:%M')
    log_entry = f"[{current_time}]: {message}\n"
    log_file.write(log_entry)
    log_file.flush()
    socketio.emit('log_update', log_entry)
    
def get_user_credentials():
    credentials = {}
    with open('credentials.txt', 'r') as f:
        for line in f:
            if line.startswith('username:'):
                credentials['username'] = line.split(':')[1].strip()
            elif line.startswith('email:'):
                credentials['email'] = line.split(':')[1].strip()
            elif line.startswith('password:'):
                credentials['password'] = line.split(':')[1].strip()
    return credentials

def get_receiver_email():
    if os.path.exists(CREDENTIALS_FILE):
        with open(CREDENTIALS_FILE, 'r') as file:
            for line in file:
                if line.startswith('email:'):
                    receiver_email = line.split(':')[1].strip()
                    return receiver_email
    return None


last_email_time = datetime.min  # Initialize the last email time
email_interval = timedelta(seconds=10)  # Set email interval to 5 seconds


def send_email_with_image(image_path):
    def send_email():
        global last_email_time
        if datetime.now() - last_email_time < email_interval:
            #log_message("Email not sent to avoid spamming")
            return

        receiver_email = get_receiver_email()
        if receiver_email is None:
            #log_message("Receiver email not found in credentials file.")
            return

        newMessage = EmailMessage()
        newMessage['Subject'] = "Theta Security System Captured Image"
        newMessage['From'] = Sender_Email
        newMessage['To'] = receiver_email
        newMessage.set_content('Let me know what you think. Image attached!')

        with open(image_path, 'rb') as f:
            image_data = f.read()
            image_name = os.path.basename(image_path)

        newMessage.add_attachment(image_data, maintype='image', subtype=imghdr.what(None, image_data), filename=image_name)

        try:
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                smtp.login(Sender_Email, Password)
                smtp.send_message(newMessage)
            last_email_time = datetime.now()  # Update the last email time
            log_message(f"Email sent with image attached.")
        except Exception as e:
            log_message(f"Failed to send email: {e}")

    email_thread = threading.Thread(target=send_email)
    email_thread.start()
    
def handle_image_tasks():
    image_counter = 0  # Initialize the image counter
    while True:
        face_img, date_folder, current_time = image_queue.get()
        if face_img is None:
            break
        image_counter += 1  # Increment the counter
        if image_counter % 10 == 0:  # Only process every 5th image
            try:
                # Save the unknown face
                face_filename = os.path.join(date_folder, f"unknown_face_{current_time}.jpg")
                cv2.imwrite(face_filename, face_img)
                # Wait for any new image event to be set
                new_image_event.wait(2)
                new_image_event.clear()
                # Check if the queue is empty, meaning this is the latest image
                if image_queue.empty():
                    # Send email with the image
                    send_email_with_image(face_filename)
            except Exception as e:
                print(f"Failed to process image task: {e}")
            finally:
                image_queue.task_done()

unknown_faces_dir = 'unknown_faces'
if not os.path.exists(unknown_faces_dir):
    os.makedirs(unknown_faces_dir)

motion_detected_dir = 'motion_detected_images'
if not os.path.exists(motion_detected_dir):
    os.makedirs(motion_detected_dir)

data_dir = 'data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Define relative paths
data_dir = 'data'
unknown_faces_dir = 'unknown_faces'
motion_detected_dir = 'motion_detected_images'
datasets_dir = os.path.join('datasets')
logs_dir = 'logs'

# Ensure directories exist
for directory in [data_dir, unknown_faces_dir, motion_detected_dir, datasets_dir, logs_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

id_name_dict = {}
id_name_file_path = os.path.join(data_dir, "id_name.txt")
if os.path.exists(id_name_file_path):
    with open(id_name_file_path, "r") as f:
        for line in f:
            id, name = line.strip().split(',')
            id_name_dict[int(id)] = name
#else:
    #log_message("id_name.txt not found, starting with an empty id-name dictionary")

video = cv2.VideoCapture(0)
backSub = cv2.createBackgroundSubtractorMOG2()
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

trainer_file_path = os.path.join(data_dir, 'Trainer.yml')
if os.path.exists(trainer_file_path):
    recognizer.read(trainer_file_path)
#else:
    #log_message("Trainer.yml not found, starting without pre-trained recognizer")

worker_thread = threading.Thread(target=handle_image_tasks)
worker_thread.start()

collecting_faces = False
collect_id = None
collect_name = None
collect_count = 0
max_collect_count = 150


@app.route('/add-email', methods=['GET', 'POST'])
def add_email():
    if request.method == 'POST':
        if request.is_json:
            # Handle JSON request
            data = request.get_json()
            new_email = data.get('email')
        else:
            # Handle form submission
            new_email = request.form.get('newEmail')

        if new_email:
            # Check if the email already exists in credentials.txt
            credentials = get_user_credentials()
            if 'email' in credentials and new_email in credentials['email']:
                if not request.is_json:
                    flash('Email already registered. Cannot add duplicate email.', 'danger')
                    return redirect(url_for('index'))
                else:
                    return jsonify({'success': False, 'message': 'Email already registered. Cannot add duplicate email.'}), 400

            # Read existing credentials from file
            with open(CREDENTIALS_FILE, 'r') as file:
                lines = file.readlines()

            # Find the line starting with 'email:'
            updated_credentials = []
            email_updated = False
            for line in lines:
                if line.strip().startswith('email:'):
                    line = f"email:{line.strip().split(':')[1].strip()}, {new_email}\n"
                    email_updated = True
                updated_credentials.append(line)

            if not email_updated:
                updated_credentials.append(f"email: {new_email}\n")

            # Write updated credentials back to file
            with open(CREDENTIALS_FILE, 'w') as file:
                file.writelines(updated_credentials)

            if not request.is_json:
                flash(f"New email '{new_email}' added successfully!", 'success')
                return redirect(url_for('index'))
            else:
                return jsonify({'success': True})
        else:
            if not request.is_json:
                flash('Failed to add email. No email provided.', 'danger')
                return redirect(url_for('index'))
            else:
                return jsonify({'success': False, 'message': 'No email provided.'}), 400
    else:
        return redirect(url_for('index'))



def is_time_in_range(start, end, current):
    if start <= end:
        return start <= current <= end
    else:
        return current >= start or current <= end

def augment_image(image, alpha=1.0, beta=0):
    """
    Adjust the brightness and contrast of an image.
    alpha: contrast control (1.0-3.0)
    beta: brightness control (0-100)
    """
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def save_augmented_faces(face_img, collect_id, collect_count):
    variations = [
        {"alpha": 1.0, "beta": 0},     # original
        {"alpha": 1.2, "beta": -10},   # brighter
        {"alpha": 0.6, "beta": -10},   # darker
    ]
    for i, var in enumerate(variations):
        augmented_img = augment_image(face_img, alpha=var["alpha"], beta=var["beta"])
        collected_image_path = os.path.join(datasets_dir, f'User.{collect_id}.{collect_count + i}.jpg')
        cv2.imwrite(collected_image_path, augmented_img)
        log_message(f"Collected augmented image: {collected_image_path}")


def generate_frames():
    global video, collecting_faces, collect_id, collect_name, collect_count
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    save_start_time = datetime.strptime('18:00', '%H:%M').time()  # 6:00 PM
    save_end_time = datetime.strptime('06:00', '%H:%M').time()    # 6:00 AM

    while True:
        ret, frame = video.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1280, 720))  # Resize frame to 1280x720

        fgMask = backSub.apply(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=(30, 30))

        nonZeroCount = cv2.countNonZero(fgMask)

        current_time = datetime.now().time()
        if nonZeroCount > 5000:
            if is_time_in_range(save_start_time, save_end_time, current_time):
                if datetime.now() - last_print_times["motion"] > print_interval:
                    log_message("Motion Detected")
                    last_print_times["motion"] = datetime.now()
                    current_date = datetime.now().strftime('%Y-%m-%d')
                    date_folder = os.path.join(motion_detected_dir, current_date)
                    if not os.path.exists(date_folder):
                        os.makedirs(date_folder)
                    motion_image_path = os.path.join(date_folder, f"motion_{datetime.now().strftime('%H-%M-%S')}.jpg")
                    cv2.imwrite(motion_image_path, frame)
                    send_email_with_image(motion_image_path)

        for (x, y, w, h) in faces:
            if collecting_faces and collect_count < max_collect_count:
                face_img = gray[y:y+h, x:x+w]
                save_augmented_faces(face_img, collect_id, collect_count)
                collect_count += 3  # Increment by 3 since we saved 3 variations
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                if collect_count >= max_collect_count:
                    with open(id_name_file_path, "a") as file:
                        file.write(f"{collect_id},{collect_name}\n")
                    log_message(f"Finished collecting faces for ID: {collect_id}, Name: {collect_name}")
                    collecting_faces = False
                    collect_id = None
                    collect_name = None
                    collect_count = 0

            else:
                if os.path.exists(trainer_file_path):
                    serial, conf = recognizer.predict(gray[y:y+h, x:x+w])
                    text = f"Unknown"
                    if conf <= 50:
                        name = id_name_dict.get(serial, "Unknown")
                        text = f"{name}"
                        if datetime.now() - last_print_times["face"] > print_interval:
                            log_message(f"Face Detected: {name}")
                            last_print_times["face"] = datetime.now()
                    else:
                        current_date = datetime.now().strftime('%Y-%m-%d')
                        date_folder = os.path.join(unknown_faces_dir, current_date)
                        if not os.path.exists(date_folder):
                            os.makedirs(date_folder)

                        face_img = frame[y:y+h, x:x+w]
                        image_queue.put((face_img, date_folder, datetime.now().strftime('%m-%d-%Y_%H-%M')))
                        new_image_event.set()
                        if datetime.now() - last_print_times["unknown_face"] > print_interval:
                            log_message("Face Detected: Unknown")
                            last_print_times["unknown_face"] = datetime.now()

                    cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)
                else:
                    cv2.putText(frame, "Recognizer model not trained yet", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



@app.route('/')
def index():
    if 'username' in session:
        registered_faces_count = len(id_name_dict)
        registered_faces = id_name_dict.items()
        return render_template('dashboard.html', registered_faces_count=registered_faces_count, registered_faces=registered_faces)
    else:
        return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user_credentials = get_user_credentials()
        
        if username == user_credentials['username'] and bcrypt.checkpw(password.encode('utf-8'), user_credentials['password'].encode('utf-8')):
            session['username'] = username
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('login.html')

def check_credentials(username, password):
    # Load credentials from credentials.txt and compare
    with open('credentials.txt', 'r') as file:
        for line in file:
            if line.strip().startswith('username:'):
                saved_username = line.strip().split(':')[-1].strip()
            elif line.strip().startswith('password:'):
                hashed_password = line.strip().split(':')[-1].strip()

    # Check if username exists and verify password
    if username == saved_username:
        return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))

    return False

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('Logged out successfully!', 'success')
    return redirect(url_for('login'))

@app.route('/collect_faces', methods=['POST'])
def collect_faces():
    global collecting_faces, collect_id, collect_name
    collect_id = int(request.form['id'])
    collect_name = request.form['name']
    collecting_faces = True
    return redirect(url_for('index'))

@app.route('/delete_user', methods=['POST'])
def delete_user():
    user_id = int(request.form['id'])
    user_name = id_name_dict.pop(user_id, None)

    if user_name:
        # Remove user from id_name.txt
        with open(id_name_file_path, "r") as f:
            lines = f.readlines()
        with open(id_name_file_path, "w") as f:
            for line in lines:
                if line.strip().split(',')[0] != str(user_id):
                    f.write(line)

        # Remove user's dataset images
        for root, dirs, files in os.walk(datasets_dir):
            for file in files:
                if file.startswith(f'User.{user_id}.'):
                    os.remove(os.path.join(root, file))

        log_message(f"Deleted user ID: {user_id}, Name: {user_name}")

        # Delete the Trainer.yml file
        if os.path.exists(trainer_file_path):
            os.remove(trainer_file_path)
            #log_message("Trainer.yml deleted")

        # Retrain the recognizer
        train_faces()

    else:
        log_message(f"Attempted to delete non-existing user ID: {user_id}")

    return redirect(url_for('index'))

@app.route('/train_faces', methods=['POST'])
def train_faces():
    face_samples = []
    ids = []

    for root, dirs, files in os.walk(datasets_dir):
        for file in files:
            if file.endswith('.jpg'):
                image_path = os.path.join(root, file)
                img = Image.open(image_path).convert('L')
                img_numpy = np.array(img, 'uint8')
                id = int(os.path.split(image_path)[-1].split(".")[1])
                face_samples.append(img_numpy)
                ids.append(id)

    if len(face_samples) < 2 or len(ids) < 2:
        log_message("Insufficient data for training. Need at least two samples.")
        flash("Insufficient data for training. Need at least two samples.", 'error')
        return redirect(url_for('index'))

    ids = np.array(ids)
    
    recognizer.train(face_samples, ids)
    
    recognizer.write(trainer_file_path)
    log_message("Training completed successfully")
    
    recognizer.read(trainer_file_path)
    restart_flask_app()
    return redirect(url_for('index'))


@app.route('/get_logs')
def get_logs():
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    log_files = [file for file in os.listdir(logs_dir) if file.endswith('.txt')]
    return jsonify(log_files)

@app.route('/get_logs/<filename>')
def get_logs_content(filename):
    file_path = os.path.join(logs_dir, filename)
    
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        return "File not found"

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/dashboard')
def dashboard():
    if 'username' in session:
        registered_faces_count = len(id_name_dict)
        registered_faces = id_name_dict.items()
        return render_template('dashboard.html', registered_faces_count=registered_faces_count, registered_faces=registered_faces)
    else:
        return redirect(url_for('login'))
    
    
@app.route('/change_password', methods=['POST'])
def change_password():
    username = request.json.get('username')
    old_password = request.json.get('oldPassword')
    new_password = request.json.get('newPassword')

    credentials_file_path = 'credentials.txt'
    credentials = {}

    # Read the credentials from the file
    with open(credentials_file_path, 'r') as file:
        for line in file:
            key, value = line.strip().split(':', 1)
            credentials[key] = value.strip()

    # Check if the username exists and verify the old password
    if 'username' in credentials and credentials['username'] == username:
        if 'password' in credentials and bcrypt.checkpw(old_password.encode('utf-8'), credentials['password'].encode('utf-8')):
            # Hash the new password
            new_password_hash = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

            # Update the password in the credentials dictionary
            credentials['password'] = new_password_hash

            # Write the updated credentials back to the file
            with open(credentials_file_path, 'w') as file:
                for key, value in credentials.items():
                    file.write(f"{key}:{value}\n")

            return jsonify({'success': True, 'message': 'Password changed successfully.'}), 200
        else:
            return jsonify({'success': False, 'message': 'Incorrect old password.'}), 400
    else:
        return jsonify({'success': False, 'message': 'Username not found.'}), 400


@app.route('/change_email', methods=['POST'])
def change_email():
    data = request.get_json()
    current_email = data.get('currentEmail')
    new_email = data.get('newEmail')

    if not current_email or not new_email:
        return jsonify({"success": False, "message": "Invalid input."}), 400

    if not os.path.exists(CREDENTIALS_FILE):
        return jsonify({"success": False, "message": "Credentials file not found."}), 404

    with open(CREDENTIALS_FILE, 'r') as file:
        content = file.read().strip()

    try:
        saved_email, saved_password_hash = content.split(',')
    except ValueError:
        return jsonify({"success": False, "message": "Invalid credentials format."}), 500

    if current_email != saved_email:
        return jsonify({"success": False, "message": "Current email is incorrect."}), 400

    new_content = f"{new_email},{saved_password_hash}"
    with open(CREDENTIALS_FILE, 'w') as file:
        file.write(new_content)

    return jsonify({"success": True, "message": "Email changed successfully."})

def read_credentials():
    with open('credentials.txt', 'r') as f:
        lines = f.readlines()
    
    credentials = {}
    for line in lines:
        key, value = line.strip().split(':', 1)
        credentials[key] = value.strip() if key != 'email' else [email.strip() for email in value.split(',')]
    return credentials

def write_credentials(credentials):
    with open('credentials.txt', 'w') as f:
        for key, value in credentials.items():
            if key == 'email':
                f.write(f"{key}:{', '.join(value)}\n")
            else:
                f.write(f"{key}:{value}\n")
                
@app.route('/delete_email', methods=['POST'])
def delete_email():
    data = request.get_json()
    email_to_delete = data.get('emailToDelete')
    
    credentials = read_credentials()
    if email_to_delete in credentials['email']:
        credentials['email'].remove(email_to_delete)
        write_credentials(credentials)
        return jsonify({"success": True, "message": "Email deleted successfully."})
    else:
        return jsonify({"success": False, "message": "Email not found."})

# Function to count occurrences of specific log entries
def count_log_entries(log_content, entry_type):
    return log_content.lower().count(entry_type.lower())

# Function to get all images associated with a log file date
def get_images_for_log(date_str):
    unknown_faces_path = os.path.join(unknown_faces_dir, date_str)
    motion_detected_path = os.path.join(motion_detected_dir, date_str)
    images = []

    if os.path.exists(unknown_faces_path):
        for file in os.listdir(unknown_faces_path):
            if file.endswith(".jpg"):
                images.append(os.path.join(unknown_faces_path, file))

    if os.path.exists(motion_detected_path):
        for file in os.listdir(motion_detected_path):
            if file.endswith(".jpg"):
                images.append(os.path.join(motion_detected_path, file))
                
    return images

def create_pdf_report(log_filename, unknown_faces_images, motion_detected_images, unknown_faces_count, motion_detected_count):
    log_file_path = os.path.join(logs_dir, log_filename)
    date_str = log_filename.replace('.txt', '')
    current_datetime_str = datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")
    
    # Retrieve the username from session
    username = session['username'] if 'username' in session else 'Guest'

    if not os.path.exists(log_file_path):
        return None

    pdf = FPDF()
    pdf.add_page()
    
    # Set font
    pdf.set_font("Arial", size=12)

    # Title
    pdf.cell(200, 10, txt=f"Security Report for {date_str}", ln=True, align='C')
    pdf.ln(10)

    # Add summary
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Number of Unknown Faces Detected: {unknown_faces_count}", ln=True)
    pdf.cell(200, 10, txt=f"Number of Motion Events Detected: {motion_detected_count}", ln=True)
    pdf.ln(5)  # Add spacing
    pdf.cell(200, 10, txt=f"Generated on: {current_datetime_str}", ln=True)
    pdf.cell(200, 10, txt=f"Generated by: {username}", ln=True)  # Add username
    pdf.ln(5)  # Add spacing
    
    # Section for Unknown Faces
    pdf.set_font("Arial", size=14, style='B')
    pdf.cell(200, 10, txt="Unknown Faces Detected", ln=True)
    pdf.ln(5)
    
    # Add table header
    pdf.set_font("Arial", size=11, style='B')
    pdf.cell(65, 10, txt="Image", border=1, align='C')
    pdf.cell(125, 10, txt="Details", border=1, align='C')
    pdf.ln()
    
    # Add images and details
    for img_path in unknown_faces_images:
        pdf.cell(65, 60, border=1)
        x = pdf.get_x() - 65  # move x back to the start of the cell
        y = pdf.get_y()
        pdf.image(img_path, x=x + 2.5, y=y + 2.5, w=60 - 5, h=60 - 5)  # Adjust margins for image fit
        pdf.set_xy(x + 65, y)  # Move to the next cell position
        pdf.cell(125, 60, txt=img_path, border=1)
        pdf.ln()

    pdf.ln(10)

    # Section for Motion Detected
    pdf.set_font("Arial", size=14, style='B')
    pdf.cell(200, 10, txt="Motion Detected Images", ln=True)
    pdf.ln(5)
    
    # Add table header
    pdf.set_font("Arial", size=11, style='B')
    pdf.cell(65, 10, txt="Image", border=1, align='C')
    pdf.cell(125, 10, txt="Details", border=1, align='C')
    pdf.ln()
    
    # Add images and details
    for img_path in motion_detected_images:
        pdf.cell(65, 60, border=1)
        x = pdf.get_x() - 65  # move x back to the start of the cell
        y = pdf.get_y()
        pdf.image(img_path, x=x + 2.5, y=y + 2.5, w=60 - 5, h=60 - 5)  # Adjust margins for image fit
        pdf.set_xy(x + 65, y)  # Move to the next cell position
        pdf.cell(125, 60, txt=img_path, border=1)
        pdf.ln()
        
    pdf.ln(10)

    output_pdf_path = os.path.join(logs_dir, f"{date_str} Generative Report.pdf")
    pdf.output(output_pdf_path)
    return output_pdf_path

@app.route('/get_user', methods=['GET'])
def get_user():
    if 'username' in session:
        return session['username']
    else:
        return 'Guest'

# Route to generate and download the PDF report
@app.route('/generate_report/<log_filename>', methods=['GET'])
def generate_report(log_filename):
    date_str = log_filename.replace('.txt', '')
    date_obj = datetime.strptime(date_str, "%B %d, %Y")
    formatted_date = date_obj.strftime("%Y-%m-%d")

    # Get paths and details for unknown faces and motion detected images
    unknown_faces_images = []  # List of tuples (image_path, details)
    motion_detected_images = []  # List of tuples (image_path, details)

    unknown_faces_path = os.path.join(unknown_faces_dir, formatted_date)
    motion_detected_path = os.path.join(motion_detected_dir, formatted_date)

    if os.path.exists(unknown_faces_path):
        for file in os.listdir(unknown_faces_path):
            if file.endswith(".jpg"):
                unknown_faces_images.append(os.path.join(unknown_faces_path, file))

    if os.path.exists(motion_detected_path):
        for file in os.listdir(motion_detected_path):
            if file.endswith(".jpg"):
                motion_detected_images.append(os.path.join(motion_detected_path, file))

    pdf_path = create_pdf_report(log_filename, unknown_faces_images, motion_detected_images, len(unknown_faces_images), len(motion_detected_images))
    if pdf_path:
        return send_from_directory(logs_dir, os.path.basename(pdf_path), as_attachment=True)
    else:
        return "Log file not found", 404
    
log_dir = r'logs'
unknown_faces_dir = r'unknown_faces'
motion_detected_dir = r'motion_detected_images'

def generate_weekly_report():
    current_date = datetime.now()
    start_date = current_date - timedelta(days=7)
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = current_date.strftime('%Y-%m-%d')
    current_datetime_str = current_date.strftime("%Y-%m-%d %I:%M:%S %p")
    
    # Retrieve the username from session
    username = session['username'] if 'username' in session else 'Guest'

    pdf = FPDF()
    pdf.add_page()
    
    # Set font
    pdf.set_font("Arial", size=12)

    # Title
    pdf.cell(200, 10, txt=f"Weekly Security Report for {start_date_str} to {end_date_str}", ln=True, align='C')
    pdf.ln(10)

    unknown_faces_images = []
    for single_date in (start_date + timedelta(n) for n in range(7)):
        date_str = single_date.strftime('%Y-%m-%d')
        date_folder = os.path.join(unknown_faces_dir, date_str)
        if os.path.exists(date_folder):
            unknown_faces_images.extend(glob.glob(os.path.join(date_folder, "*.jpg")))
    
    unknown_faces_count = len(unknown_faces_images)

    motion_detected_images = []
    for single_date in (start_date + timedelta(n) for n in range(7)):
        date_str = single_date.strftime('%Y-%m-%d')
        date_folder = os.path.join(motion_detected_dir, date_str)
        if os.path.exists(date_folder):
            motion_detected_images.extend(glob.glob(os.path.join(date_folder, "*.jpg")))

    motion_detected_count = len(motion_detected_images)

    # Add summary
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Number of Unknown Faces Detected: {unknown_faces_count}", ln=True)
    pdf.cell(200, 10, txt=f"Number of Motion Events Detected: {motion_detected_count}", ln=True)
    pdf.ln(5)  # Add spacing
    pdf.cell(200, 10, txt=f"Generated on: {current_datetime_str}", ln=True)
    pdf.cell(200, 10, txt=f"Generated by: {username}", ln=True)
    pdf.ln(5)  # Add spacing
    
    # Section for Unknown Faces
    pdf.set_font("Arial", size=14, style='B')
    pdf.cell(200, 10, txt="Unknown Faces Detected", ln=True)
    pdf.ln(5)

    if not unknown_faces_images:
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="No unknown faces detected in the last 7 days.", ln=True, align='L')
    else:
        # Add table header
        pdf.set_font("Arial", size=11, style='B')
        pdf.cell(65, 10, txt="Image", border=1, align='C')
        pdf.cell(125, 10, txt="Details", border=1, align='C')
        pdf.ln()

        for img_path in unknown_faces_images:
            pdf.cell(65, 60, border=1)
            x = pdf.get_x() - 65  # move x back to the start of the cell
            y = pdf.get_y()
            pdf.image(img_path, x=x + 2.5, y=y + 2.5, w=60 - 5, h=60 - 5)  # Adjust margins for image fit
            pdf.set_xy(x + 65, y)  # Move to the next cell position
            pdf.cell(125, 60, txt=img_path, border=1)
            pdf.ln()

    pdf.ln(10)

    # Section for Motion Detected
    pdf.set_font("Arial", size=14, style='B')
    pdf.cell(200, 10, txt="Motion Detected Images", ln=True)
    pdf.ln(5)

    if not motion_detected_images:
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="No motion detected events in the last 7 days.", ln=True, align='L')
    else:
        # Add table header
        pdf.set_font("Arial", size=11, style='B')
        pdf.cell(65, 10, txt="Image", border=1, align='C')
        pdf.cell(125, 10, txt="Details", border=1, align='C')
        pdf.ln()

        for img_path in motion_detected_images:
            pdf.cell(65, 60, border=1)
            x = pdf.get_x() - 65  # move x back to the start of the cell
            y = pdf.get_y()
            pdf.image(img_path, x=x + 2.5, y=y + 2.5, w=60 - 5, h=60 - 5)  # Adjust margins for image fit
            pdf.set_xy(x + 65, y)  # Move to the next cell position
            pdf.cell(125, 60, txt=img_path, border=1)
            pdf.ln()
        
    pdf.ln(10)

    report_path = os.path.join(log_dir, f"weekly_report_{end_date_str}.pdf")
    pdf.output(report_path)
    return report_path

@app.route('/weekly_report', methods=['GET'])
def weekly_report():
    if 'username' in session:
        report_path = generate_weekly_report()
        return send_from_directory(log_dir, os.path.basename(report_path), as_attachment=True)
    else:
        return redirect(url_for('login'))
    

def generate_monthly_report():
    current_date = datetime.now()
    start_date = current_date - timedelta(days=30)
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = current_date.strftime('%Y-%m-%d')
    current_datetime_str = current_date.strftime("%Y-%m-%d %I:%M:%S %p")
    
    # Retrieve the username from session
    username = session.get('username', 'Guest')

    pdf = FPDF()
    pdf.add_page()
    
    # Set font
    pdf.set_font("Arial", size=12)

    # Title
    pdf.cell(200, 10, txt=f"Monthly Security Report for {start_date_str} to {end_date_str}", ln=True, align='C')
    pdf.ln(10)

    unknown_faces_images = []
    for single_date in (start_date + timedelta(n) for n in range(30)):
        date_str = single_date.strftime('%Y-%m-%d')
        date_folder = os.path.join(unknown_faces_dir, date_str)
        if os.path.exists(date_folder):
            unknown_faces_images.extend(glob.glob(os.path.join(date_folder, "*.jpg")))
    
    unknown_faces_count = len(unknown_faces_images)

    motion_detected_images = []
    for single_date in (start_date + timedelta(n) for n in range(30)):
        date_str = single_date.strftime('%Y-%m-%d')
        date_folder = os.path.join(motion_detected_dir, date_str)
        if os.path.exists(date_folder):
            motion_detected_images.extend(glob.glob(os.path.join(date_folder, "*.jpg")))

    motion_detected_count = len(motion_detected_images)

    # Add summary
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Number of Unknown Faces Detected: {unknown_faces_count}", ln=True)
    pdf.cell(200, 10, txt=f"Number of Motion Events Detected: {motion_detected_count}", ln=True)
    pdf.ln(5)  # Add spacing
    pdf.cell(200, 10, txt=f"Generated on: {current_datetime_str}", ln=True)
    pdf.cell(200, 10, txt=f"Generated by: {username}", ln=True)
    pdf.ln(5)  # Add spacing
    
    # Section for Unknown Faces
    pdf.set_font("Arial", size=14, style='B')
    pdf.cell(200, 10, txt="Unknown Faces Detected", ln=True)
    pdf.ln(5)

    if not unknown_faces_images:
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="No unknown faces detected in the last 30 days.", ln=True, align='L')
    else:
        # Add table header
        pdf.set_font("Arial", size=11, style='B')
        pdf.cell(65, 10, txt="Image", border=1, align='C')
        pdf.cell(125, 10, txt="Details", border=1, align='C')
        pdf.ln()

        for img_path in unknown_faces_images:
            pdf.cell(65, 60, border=1)
            x = pdf.get_x() - 65  # move x back to the start of the cell
            y = pdf.get_y()
            pdf.image(img_path, x=x + 2.5, y=y + 2.5, w=60 - 5, h=60 - 5)  # Adjust margins for image fit
            pdf.set_xy(x + 65, y)  # Move to the next cell position
            pdf.cell(125, 60, txt=img_path, border=1)
            pdf.ln()

    pdf.ln(10)

    # Section for Motion Detected
    pdf.set_font("Arial", size=14, style='B')
    pdf.cell(200, 10, txt="Motion Detected Images", ln=True)
    pdf.ln(5)

    if not motion_detected_images:
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="No motion detected events in the last 30 days.", ln=True, align='L')
    else:
        # Add table header
        pdf.set_font("Arial", size=11, style='B')
        pdf.cell(65, 10, txt="Image", border=1, align='C')
        pdf.cell(125, 10, txt="Details", border=1, align='C')
        pdf.ln()

        for img_path in motion_detected_images:
            pdf.cell(65, 60, border=1)
            x = pdf.get_x() - 65  # move x back to the start of the cell
            y = pdf.get_y()
            pdf.image(img_path, x=x + 2.5, y=y + 2.5, w=60 - 5, h=60 - 5)  # Adjust margins for image fit
            pdf.set_xy(x + 65, y)  # Move to the next cell position
            pdf.cell(125, 60, txt=img_path, border=1)
            pdf.ln()
        
    pdf.ln(10)

    report_path = os.path.join(log_dir, f"monthly_report_{end_date_str}.pdf")
    pdf.output(report_path)
    return report_path


@app.route('/monthly_report', methods=['GET'])
def monthly_report():
    if 'username' in session:
        report_path = generate_monthly_report()
        return send_from_directory(log_dir, os.path.basename(report_path), as_attachment=True)
    else:
        return redirect(url_for('login'))
    
    
def generate_annual_report():
    current_date = datetime.now()
    start_date = current_date - timedelta(days=365)  # Adjust for 365 days for annual report
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = current_date.strftime('%Y-%m-%d')
    current_datetime_str = current_date.strftime("%Y-%m-%d %I:%M:%S %p")
    
    # Retrieve the username from session
    username = session.get('username', 'Guest')

    pdf = FPDF()
    pdf.add_page()
    
    # Set font
    pdf.set_font("Arial", size=12)

    # Title
    pdf.cell(200, 10, txt=f"Annual Security Report for {start_date_str} to {end_date_str}", ln=True, align='C')
    pdf.ln(10)

    unknown_faces_images = []
    for single_date in (start_date + timedelta(n) for n in range(365)):
        date_str = single_date.strftime('%Y-%m-%d')
        date_folder = os.path.join(unknown_faces_dir, date_str)
        if os.path.exists(date_folder):
            unknown_faces_images.extend(glob.glob(os.path.join(date_folder, "*.jpg")))
    
    unknown_faces_count = len(unknown_faces_images)

    motion_detected_images = []
    for single_date in (start_date + timedelta(n) for n in range(365)):
        date_str = single_date.strftime('%Y-%m-%d')
        date_folder = os.path.join(motion_detected_dir, date_str)
        if os.path.exists(date_folder):
            motion_detected_images.extend(glob.glob(os.path.join(date_folder, "*.jpg")))

    motion_detected_count = len(motion_detected_images)

    # Add summary
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Number of Unknown Faces Detected: {unknown_faces_count}", ln=True)
    pdf.cell(200, 10, txt=f"Number of Motion Events Detected: {motion_detected_count}", ln=True)
    pdf.ln(5)  # Add spacing
    pdf.cell(200, 10, txt=f"Generated on: {current_datetime_str}", ln=True)
    pdf.cell(200, 10, txt=f"Generated by: {username}", ln=True)
    pdf.ln(5)  # Add spacing
    
    # Section for Unknown Faces
    pdf.set_font("Arial", size=14, style='B')
    pdf.cell(200, 10, txt="Unknown Faces Detected", ln=True)
    pdf.ln(5)

    if not unknown_faces_images:
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="No unknown faces detected in the last year.", ln=True, align='L')
    else:
        # Add table header
        pdf.set_font("Arial", size=11, style='B')
        pdf.cell(65, 10, txt="Image", border=1, align='C')
        pdf.cell(125, 10, txt="Details", border=1, align='C')
        pdf.ln()

        for img_path in unknown_faces_images:
            pdf.cell(65, 60, border=1)
            x = pdf.get_x() - 65  # move x back to the start of the cell
            y = pdf.get_y()
            pdf.image(img_path, x=x + 2.5, y=y + 2.5, w=60 - 5, h=60 - 5)  # Adjust margins for image fit
            pdf.set_xy(x + 65, y)  # Move to the next cell position
            pdf.cell(125, 60, txt=img_path, border=1)
            pdf.ln()

    pdf.ln(10)

    # Section for Motion Detected
    pdf.set_font("Arial", size=14, style='B')
    pdf.cell(200, 10, txt="Motion Detected Images", ln=True)
    pdf.ln(5)

    if not motion_detected_images:
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="No motion detected events in the last year.", ln=True, align='L')
    else:
        # Add table header
        pdf.set_font("Arial", size=11, style='B')
        pdf.cell(65, 10, txt="Image", border=1, align='C')
        pdf.cell(125, 10, txt="Details", border=1, align='C')
        pdf.ln()

        for img_path in motion_detected_images:
            pdf.cell(65, 60, border=1)
            x = pdf.get_x() - 65  # move x back to the start of the cell
            y = pdf.get_y()
            pdf.image(img_path, x=x + 2.5, y=y + 2.5, w=60 - 5, h=60 - 5)  # Adjust margins for image fit
            pdf.set_xy(x + 65, y)  # Move to the next cell position
            pdf.cell(125, 60, txt=img_path, border=1)
            pdf.ln()
        
    pdf.ln(10)

    report_path = os.path.join(log_dir, f"annual_report_{end_date_str}.pdf")
    pdf.output(report_path)
    return report_path


@app.route('/annual_report', methods=['GET'])
def annual_report():
    if 'username' in session:
        report_path = generate_annual_report()
        return send_from_directory(log_dir, os.path.basename(report_path), as_attachment=True)
    else:
        return redirect(url_for('login'))
    
if __name__ == '__main__':
    try:
        socketio.run(app, debug=True, allow_unsafe_werkzeug=True)
    finally:
        if log_file is not None:
            log_file.close()
        image_queue.put((None, None, None))
        worker_thread.join()
        video.release()
        cv2.destroyAllWindows()