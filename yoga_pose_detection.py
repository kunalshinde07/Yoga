import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd
from scipy.signal import find_peaks, butter, filtfilt
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

# Parameters for audio input
SAMPLE_RATE = 44100  # Sampling rate in Hz
DURATION = 10        # Duration to record in seconds

# Initialize Mediapipe pose detection
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Reference landmarks for basic poses (simplified example)
POSES = {
    "Tree Pose": [0.5, 0.5, 0.6, 0.9, 0.4, 0.2, 0.5],
    "Warrior Pose": [0.5, 0.5, 0.7, 0.9, 0.3, 0.2, 0.7],
}

# Bandpass filter function for breathing signal
def bandpass_filter(signal, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

# Analyze breathing function
def analyze_breathing(audio, sample_rate):
    # Convert to mono and normalize
    audio = audio.flatten()
    audio = (audio - np.mean(audio)) / np.std(audio)
    
    # Apply bandpass filter
    filtered_signal = bandpass_filter(audio, lowcut=0.1, highcut=2.0, fs=sample_rate)
    
    # Calculate smoothed envelope
    envelope = np.abs(filtered_signal)
    smoothed = np.convolve(envelope, np.ones(500) / 500, mode='same')
    
    # Detect peaks (breath cycles)
    height_threshold = 0.1 * np.max(smoothed)
    peaks, _ = find_peaks(smoothed, distance=sample_rate // 2, height=height_threshold)
    breath_intervals = np.diff(peaks) / sample_rate  # Time between peaks in seconds
    
    # Calculate breathing rate
    avg_rate = 60 / np.mean(breath_intervals) if len(breath_intervals) > 0 else 0
    return smoothed, peaks, avg_rate, breath_intervals

# Function to classify pose
def classify_pose(landmarks):
    user_pose = [
        landmarks[mp_pose.PoseLandmark.NOSE.value].x,
        landmarks[mp_pose.PoseLandmark.NOSE.value].y,
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
        0  # Add a placeholder value for consistency
    ]

    min_distance = float("inf")
    matched_pose = "Unknown"
    for pose_name, ref_pose in POSES.items():
        distance = pairwise_distances([user_pose], [ref_pose], metric="euclidean")[0][0]
        if distance < min_distance:
            min_distance = distance
            matched_pose = pose_name
    
    return matched_pose, min_distance


# Function to provide feedback
def give_feedback(pose_name, landmarks):
    if pose_name == "Tree Pose":
        return "Keep your balance and align your hands above your head."
    elif pose_name == "Warrior Pose":
        return "Widen your stance and straighten your back leg."
    else:
        return "Try adjusting your posture to match the pose."

# Record breathing audio in a separate thread
print("Recording breathing sounds...")
audio_data = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=1, dtype='float64')
sd.wait()
print("Recording complete.")

# Start webcam feed for pose detection
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Detect pose and provide feedback
        pose_name, _ = classify_pose(results.pose_landmarks.landmark)
        feedback = give_feedback(pose_name, results.pose_landmarks.landmark)

        # Display pose feedback
        cv2.putText(frame, f"Pose: {pose_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, feedback, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Show the frame
    cv2.imshow("Pose Detection & Breathing Analysis", frame)

    # Break on 'q' key
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Analyze the recorded audio for breathing
smoothed_signal, peaks, avg_rate, intervals = analyze_breathing(audio_data, SAMPLE_RATE)
print(f"Average Breathing Rate: {avg_rate:.2f} breaths per minute")
print(f"Breath Intervals: {intervals}")

# Plot the breathing signal
time_axis = np.arange(len(smoothed_signal)) / SAMPLE_RATE
plt.figure(figsize=(10, 5))
plt.plot(time_axis, smoothed_signal, label="Smoothed Breathing Signal")
plt.scatter(time_axis[peaks], smoothed_signal[peaks], color='red', label="Detected Breaths")
plt.title("Breathing Pattern Analysis")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.legend()
plt.show()



def generate_insights(data):
    insights = {}
    
    # Average Metrics
    avg_pose_accuracy = data["Pose_Accuracy (%)"].mean()
    avg_duration = data["Session_Duration (minutes)"].mean()
    avg_engagement = data["Engagement_Score (1-10)"].mean()
    avg_mood_improvement = data["Mood_Improvement (1-10)"].mean()
    
    insights["Average Pose Accuracy"] = f"{avg_pose_accuracy:.2f}%"
    insights["Average Session Duration"] = f"{avg_duration:.2f} minutes"
    insights["Average Engagement Score"] = f"{avg_engagement:.2f}/10"
    insights["Average Mood Improvement"] = f"{avg_mood_improvement:.2f}/10"
    
    # Strengths
    insights["Strength"] = ""
    if avg_pose_accuracy > 85:
        insights["Strength"] += "Excellent pose accuracy! Keep up the great form. "
    if avg_engagement > 8:
        insights["Strength"] += "You're highly engaged; consider advanced routines. "
    
    # Improvement Areas
    insights["Improvement"] = ""
    if avg_duration < 20:
        insights["Improvement"] += "Try to increase session duration for better results. "
    if avg_mood_improvement < 5:
        insights["Improvement"] += "Focus on relaxation poses and breathwork to improve mood."
    
    # Remove empty keys
    insights = {k: v.strip() for k, v in insights.items() if v.strip()}
    return insights


# Provide sample user data
import pandas as pd

user_data = pd.DataFrame({
    "Session_ID": [1, 2, 3, 4, 5],
    "Pose_Accuracy (%)": [86, 90, 88, 83, 87],
    "Session_Duration (minutes)": [15, 18, 22, 25, 20],
    "Engagement_Score (1-10)": [7, 9, 8, 9, 8],
    "Mood_Improvement (1-10)": [4, 5, 6, 4, 5]
})

# Generate insights
user_insights = generate_insights(user_data)
for key, value in user_insights.items():
    print(f"{key}: {value}")


# Improved Visualization Function
import matplotlib.pyplot as plt

def visualize_data(data):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    
    # Pose Accuracy
    axs[0, 0].plot(data["Session_ID"], data["Pose_Accuracy (%)"], marker="o", color="blue")
    axs[0, 0].set_title("Pose Accuracy Over Sessions")
    axs[0, 0].set_xlabel("Session ID")
    axs[0, 0].set_ylabel("Pose Accuracy (%)")
    
    # Session Duration
    axs[0, 1].bar(data["Session_ID"], data["Session_Duration (minutes)"], color="green")
    axs[0, 1].set_title("Session Duration")
    axs[0, 1].set_xlabel("Session ID")
    axs[0, 1].set_ylabel("Duration (minutes)")
    
    # Engagement Score
    axs[1, 0].plot(data["Session_ID"], data["Engagement_Score (1-10)"], marker="x", color="purple")
    axs[1, 0].set_title("Engagement Score Over Sessions")
    axs[1, 0].set_xlabel("Session ID")
    axs[1, 0].set_ylabel("Engagement Score")
    
    # Mood Improvement
    axs[1, 1].scatter(data["Session_ID"], data["Mood_Improvement (1-10)"], color="orange")
    axs[1, 1].set_title("Mood Improvement Over Sessions")
    axs[1, 1].set_xlabel("Session ID")
    axs[1, 1].set_ylabel("Mood Improvement")
    
    # Add gridlines to all subplots for better readability
    for ax in axs.flat:
        ax.grid(True)

    plt.tight_layout()
    plt.show()
