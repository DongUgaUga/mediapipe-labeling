import cv2
import mediapipe as mp
import numpy as np
import os
import csv

# MediaPipe Holistic 설정
mp_holistic = mp.solutions.holistic

# 지원 확장자
SUPPORTED_EXTS = ('.mp4', '.avi', '.mts', '.MTS', '.mov', '.MOV')

# 좌표 추출
def extract_holistic_landmarks(video_path):
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(video_path)
    all_landmarks = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)

        frame_landmarks = []

        # POSE: 33개
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                frame_landmarks.extend([lm.x, lm.y, lm.z])
        else:
            frame_landmarks.extend([0.0] * 33 * 3)

        # LEFT HAND: 21개
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                frame_landmarks.extend([lm.x, lm.y, lm.z])
        else:
            frame_landmarks.extend([0.0] * 21 * 3)

        # RIGHT HAND: 21개
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                frame_landmarks.extend([lm.x, lm.y, lm.z])
        else:
            frame_landmarks.extend([0.0] * 21 * 3)

        all_landmarks.append(frame_landmarks)

    cap.release()
    holistic.close()
    return np.array(all_landmarks)

# 전체 폴더 순회 및 저장
def process_all_videos(root_folder, output_folder, label_csv_path):
    os.makedirs(output_folder, exist_ok=True)
    label_records = []
    existing_files = set(os.listdir(output_folder))

    for subfolder in sorted(os.listdir(root_folder)):
        sub_path = os.path.join(root_folder, subfolder)
        if not os.path.isdir(sub_path):
            continue

        for file_name in os.listdir(sub_path):
            if file_name.endswith(SUPPORTED_EXTS):
                video_path = os.path.join(sub_path, file_name)

                # 고유 이름 만들기
                video_id = os.path.splitext(file_name)[0]
                save_name = f"{video_id}.npy"
                save_path = os.path.join(output_folder, save_name)

                # 이미 처리된 파일은 건너뜀
                if save_name in existing_files:
                    print(f"⏭ 이미 존재함, 건너뜀: {save_name}")
                    continue

                try:
                    print(f"▶ 처리 중: {video_path}")
                    landmarks = extract_holistic_landmarks(video_path)
                    np.save(save_path, landmarks)

                    numeric_id = ''.join(filter(str.isdigit, video_id))
                    label_records.append([save_name, numeric_id])
                    print(f"✔ 저장 완료: {save_name}")

                except Exception as e:
                    print(f"❌ 실패: {video_path} - {e}")

    # CSV 저장
    with open(label_csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        if os.stat(label_csv_path).st_size == 0:
            writer.writerow(["file_name", "label_id"])
        writer.writerows(label_records)
    print(f"📄 라벨 CSV 저장 완료: {label_csv_path}")

process_all_videos(
    root_folder="/mnt/hdd/woo/수어 영상",
    output_folder="./processed/npy",
    label_csv_path="./processed/labels.csv"
)