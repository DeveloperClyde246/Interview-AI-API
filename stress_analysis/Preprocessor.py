import cv2
import numpy as np

class Preprocessor:
    def __init__(self):
        self.data = None

    def load_and_process_data(self, data_path):
        images = []
        labels = []
        image = cv2.imread(data_path)
        images.append(image)
        labels.append(0)

        self.data = images
        self.convert_to_grayscale()
        self.resize_image((48, 48))
        self.histogram_equalization()
        self.normalize()
        images = self.data
        
        images = np.array(images, dtype='float32')
        labels = np.array(labels)
        
        return images, labels
    
    def load_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        success = True
        count = 0
        while success:
            success, frame = cap.read()
            if not success or frame is None:
                print("Video fully processed. All frames extracted.")
                break
            frames.append(frame)
            count += 1
        cap.release()
        print(count)
        self.data = frames
        return self
    
    def detect_and_crop_face(self):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
        cropped_faces = []
        for frame in self.data:
            faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)
            for (x, y, w, h) in faces:
                cropped_face = frame[y:y+h, x:x+w]
                cropped_faces.append(cropped_face)
        self.data = cropped_faces
        return self

    def detect_and_crop_eye(self):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        cropped_eyes = []
        for frame in self.data:
            faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)
            for (x, y, w, h) in faces:
                roi_gray = frame[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
                if len(eyes) > 0:
                    # Sort eyes by x-coordinate to get the right eye
                    eyes = sorted(eyes, key=lambda e: e[0], reverse=True)
                    right_eye = eyes[0]
                    (ex, ey, ew, eh) = right_eye
                    cropped_eye = roi_gray[ey:ey+eh, ex:ex+ew]
                    cropped_eyes.append(cropped_eye)
        self.data = cropped_eyes
        return self
    
    def resize_image(self, target_size):
        resized_images = []
        for image in self.data:
            resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
            resized_images.append(resized_image)
        self.data = resized_images
        return self

    def convert_to_grayscale(self):
        grayscale_images = []
        for image in self.data:
            grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            grayscale_images.append(grayscale_image)
        self.data = grayscale_images
        return self

    def histogram_equalization(self):
        equalized_images = []
        for image in self.data:
            equalized_image = cv2.equalizeHist(image)
            equalized_images.append(equalized_image)
        self.data = equalized_images
        return self

    def normalize(self):
        normalized_images = []
        for image in self.data:
            normalized_image = image / 255.0
            normalized_images.append(normalized_image)
        self.data = normalized_images
        return self

    def preprocessFER(self,video_path):
        self.load_video(video_path)
        self.convert_to_grayscale()
        self.detect_and_crop_face()
        self.resize_image((48, 48))
        #self.histogram_equalization()
        self.normalize()
        return self.data
    
    def preprocessEyeGaze(self,video_path):
        self.load_video(video_path)
        self.convert_to_grayscale()
        self.detect_and_crop_eye()
        self.resize_image((60, 60))
        self.normalize()
        return self.data
    
    def get_video_duration(video_path):
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)  # Frames per second
        frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)  # Total number of frames
        duration = frame_count / fps  # Duration in seconds
        video.release()
        return duration
    
    def get_data(self):
        return self.data