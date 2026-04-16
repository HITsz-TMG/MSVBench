import cv2
import numpy as np
import os

class VideoQualityAnalyzer:
    def __init__(self):
        pass
    
    def get_brightness(self, image):
        """Calculate the average brightness of an image."""
        if len(image.shape) == 3:
            gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = image
        return np.mean(gray_img)

    def get_contrast(self, image):
        """Calculate image contrast (standard deviation of grayscale intensity)."""
        if len(image.shape) == 3:
            gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = image
        return np.std(gray_img)

    def get_saturation(self, image):
        """Calculate the average saturation of an image."""
        if len(image.shape) == 3:
            hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            s_channel = hsv_img[:, :, 1]
            return np.mean(s_channel)
        else:
            # Saturation is zero for grayscale images.
            return 0.0

    def extract_frame_at_second(self, video_path, second):
        """Extract a frame at the specified second from a video."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(second * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            return frame
        else:
            print(f"Error: Failed to extract frame at second {second}")
            return None

    def get_video_duration(self, video_path):
        """Get video duration in seconds."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps
        cap.release()
        return duration

    def get_last_frame(self, video_path):
        """Get the last frame of a video."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return None
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            return frame
        else:
            print("Error: Failed to extract the last frame")
            return None

    def get_first_frame(self, video_path):
        """Get the first frame of a video."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return None
        
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            return frame
        else:
            print("Error: Failed to extract the first frame")
            return None

    def analyze_frame_metrics(self, frame):
        """Analyze brightness, contrast, and saturation for a single frame."""
        if frame is None:
            return None
        
        return {
            'brightness': self.get_brightness(frame),
            'contrast': self.get_contrast(frame),
            'saturation': self.get_saturation(frame)
        }

    def calculate_metric_changes(self, metrics1, metrics2):
        """Calculate the change between two sets of metrics."""
        if metrics1 is None or metrics2 is None:
            return None
        
        changes = {
            'brightness_change': abs(metrics2['brightness'] - metrics1['brightness']),
            'contrast_change': abs(metrics2['contrast'] - metrics1['contrast']),
            'saturation_change': abs(metrics2['saturation'] - metrics1['saturation'])
        }
        
        # Sum the three metric changes.
        total_change = changes['brightness_change'] + changes['contrast_change'] + changes['saturation_change']
        changes['total_change'] = total_change
        
        return changes

    def analyze(self, video_path1, video_path2=None):
        """
        Analyze video quality metrics.
        
        Args:
            video_path1: Path to the first video.
            video_path2: Path to the second video (optional).
            
        Returns:
            dict: A dictionary containing analysis results and score.
        """
        if video_path2 is None:
            # Single-video analysis: compute inter-frame changes within one video.
            return self._analyze_single_video(video_path1)
        else:
            # Two-video analysis: last frame of first video vs first frame of second.
            return self._analyze_two_videos(video_path1, video_path2)

    def _analyze_single_video(self, video_path):
        """Analyze inter-frame changes within a single video."""
        print(f"Analyzing single video: {video_path}")
        
        if not os.path.exists(video_path):
            return {"error": f"Video file does not exist: {video_path}"}
        
        duration = self.get_video_duration(video_path)
        if duration < 2:
            return {"error": "Video duration is less than 2 seconds; cannot perform 1-second interval analysis"}
        
        all_changes = []
        
        # Extract one frame per second and compare adjacent seconds.
        for second in range(int(duration) - 1):
            frame1 = self.extract_frame_at_second(video_path, second)
            frame2 = self.extract_frame_at_second(video_path, second + 1)
            
            if frame1 is not None and frame2 is not None:
                metrics1 = self.analyze_frame_metrics(frame1)
                metrics2 = self.analyze_frame_metrics(frame2)
                
                changes = self.calculate_metric_changes(metrics1, metrics2)
                if changes is not None:
                    all_changes.append(changes['total_change'])
                    print(f"Change from second {second} to {second+1}: {changes['total_change']:.2f}")
        
        if not all_changes:
            return {"error": "Unable to extract valid inter-frame change data"}
        
        # Compute the average change across all sampled intervals.
        avg_change = np.mean(all_changes)
        score = max(0, 100 - avg_change)
        
        return {
            "type": "single_video",
            "video_path": video_path,
            "frame_changes": all_changes,
            "average_change": avg_change,
            "score": score
        }

    def _analyze_two_videos(self, video_path1, video_path2):
        """Analyze changes between two videos."""
        print(f"Analyzing two videos: {video_path1} -> {video_path2}")
        
        if not os.path.exists(video_path1):
            return {"error": f"First video file does not exist: {video_path1}"}
        if not os.path.exists(video_path2):
            return {"error": f"Second video file does not exist: {video_path2}"}
        
        # Get the last frame from the first video.
        last_frame = self.get_last_frame(video_path1)
        # Get the first frame from the second video.
        first_frame = self.get_first_frame(video_path2)
        
        if last_frame is None or first_frame is None:
            return {"error": "Failed to extract video frames"}
        
        metrics1 = self.analyze_frame_metrics(last_frame)
        metrics2 = self.analyze_frame_metrics(first_frame)
        
        changes = self.calculate_metric_changes(metrics1, metrics2)
        if changes is None:
            return {"error": "Failed to calculate metric changes"}
        
        score = max(0, 100 - changes['total_change'])
        
        return {
            "type": "two_videos",
            "video_path1": video_path1,
            "video_path2": video_path2,
            "metrics1": metrics1,
            "metrics2": metrics2,
            "changes": changes,
            "score": score
        }

if __name__ == "__main__":
    analyzer = VideoQualityAnalyzer()
    
    # Example: compute internal style consistency for a single video.
    video_path = "path/to/your/video.mp4"  # Replace with an actual video path.
    internal_result = analyzer.analyze(video_path)
    if "error" not in internal_result:
        print(f'Internal style consistency score: {internal_result["score"]:.4f}')
    else:
        print(f'Error: {internal_result["error"]}')
    
    # Example: compute style consistency between two videos.
    video_path1 = "path/to/your/video1.mp4"  # Replace with an actual video path.
    video_path2 = "path/to/your/video2.mp4"  # Replace with an actual video path.
    cross_result = analyzer.analyze(video_path1, video_path2)
    if "error" not in cross_result:
        print(f'Cross-video style consistency score: {cross_result["score"]:.4f}')
    else:
        print(f'Error: {cross_result["error"]}')