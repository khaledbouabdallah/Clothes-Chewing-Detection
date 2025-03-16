import cv2
import mediapipe as mp
import numpy as np
import time
import winsound
from collections import deque

class EdgeMouthDetector:
    # MediaPipe initialization
    mp_face_mesh = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # Constants
    MOUTH_LANDMARKS = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
    ALERT_COOLDOWN = 3  # Seconds between alerts
    TIME_THRESHOLD = 1.5  # Seconds of sustained detection before alert
    MOUTH_REGION_DILATION = 15  # Pixels to dilate mouth region
    ALERT_MARGIN_FACTOR = 1.8  # Multiplier for alert area relative to mouth size
    EDGE_THRESHOLD_LOW = 50  # Canny edge detection low threshold
    EDGE_THRESHOLD_HIGH = 150  # Canny edge detection high threshold
    OVERLAP_THRESHOLD = 0.15  # Percentage of mouth covered by edges to trigger alert
    
    def __init__(self):
        # Initialize detectors
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.hands = self.mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_hands=2
        )
        
        # State variables
        self.last_alert_time = 0
        self.object_detected_start_time = 0
        self.object_detected_active = False
        
        # Initialize the camera
        self.cap = None
        
        # History for mouth coverage calculations
        self.coverage_history = deque(maxlen=5)  # Reduced history length for faster response
        
        # Debug flags
        self.debug_mode = True
    
    def open_camera(self, camera_index=0):
        """Open the camera."""
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            print(f"Could not open camera at index {camera_index}")
            return False
        return True
    
    def detect_mouth(self, face_landmarks, image_shape):
        """Extract mouth landmarks and create mouth and alert region masks."""
        h, w = image_shape[:2]
        mouth_points = []
        mouth_center = None
        mouth_mask = np.zeros((h, w), dtype=np.uint8)
        alert_mask = np.zeros((h, w), dtype=np.uint8)
        mouth_roi = None
        face_size = 0
        
        if not face_landmarks:
            return [], None, mouth_mask, alert_mask, mouth_roi, face_size
        
        # Get outer mouth points
        for idx in self.MOUTH_LANDMARKS:
            landmark = face_landmarks.landmark[idx]
            x, y = int(landmark.x * w), int(landmark.y * h)
            mouth_points.append((x, y))
        
        # Calculate face size for relative scaling
        # Use distance between eyes as a reference
        left_eye = face_landmarks.landmark[33]  # Left eye outer corner
        right_eye = face_landmarks.landmark[263]  # Right eye outer corner
        face_size = np.sqrt((right_eye.x - left_eye.x)**2 + (right_eye.y - left_eye.y)**2) * w
        
        if mouth_points:
            # Calculate mouth center
            x_coordinates = [p[0] for p in mouth_points]
            y_coordinates = [p[1] for p in mouth_points]
            
            mouth_center = (
                sum(x_coordinates) / len(x_coordinates),
                sum(y_coordinates) / len(y_coordinates)
            )
            
            # Calculate mouth bounding box
            min_x = max(0, min(x_coordinates))
            min_y = max(0, min(y_coordinates))
            max_x = min(w, max(x_coordinates))
            max_y = min(h, max(y_coordinates))
            
            # Calculate mouth width and height
            mouth_width = max_x - min_x
            mouth_height = max_y - min_y
            
            # Extract ROI for mouth (with margin)
            margin = int(max(mouth_width, mouth_height) * 0.2)
            roi_min_x = max(0, min_x - margin)
            roi_min_y = max(0, min_y - margin)
            roi_max_x = min(w, max_x + margin)
            roi_max_y = min(h, max_y + margin)
            
            if roi_min_x < roi_max_x and roi_min_y < roi_max_y:
                mouth_roi = (roi_min_x, roi_min_y, roi_max_x, roi_max_y)
            
            # Create a mask for the mouth region
            mouth_np_points = np.array(mouth_points, dtype=np.int32)
            cv2.fillPoly(mouth_mask, [mouth_np_points], 255)
            
            # Dilate the mask to create a region around the mouth
            mouth_dilation = self.MOUTH_REGION_DILATION
            if face_size > 0:
                # Scale dilation based on face size
                mouth_dilation = max(5, int(face_size * 0.05))  # 5% of face width, minimum 5px
            
            kernel = np.ones((mouth_dilation, mouth_dilation), np.uint8)
            mouth_mask = cv2.dilate(mouth_mask, kernel, iterations=1)
            
            # Create a larger alert area mask based on face size
            alert_dilation = int(mouth_dilation * self.ALERT_MARGIN_FACTOR)
            alert_kernel = np.ones((alert_dilation, alert_dilation), np.uint8)
            alert_mask = cv2.dilate(mouth_mask, alert_kernel, iterations=1)
        
        return mouth_points, mouth_center, mouth_mask, alert_mask, mouth_roi, face_size
    
    def create_hand_mask(self, hand_landmarks, image_shape, face_size=0):
        """Create a mask of hand regions."""
        h, w = image_shape[:2]
        hand_mask = np.zeros((h, w), dtype=np.uint8)
        
        if not hand_landmarks:
            return hand_mask
        
        # Scale hand mask thickness based on face size
        hand_point_radius = 30
        hand_line_thickness = 20
        if face_size > 0:
            hand_point_radius = max(20, int(face_size * 0.1))  # 10% of face width, minimum 20px
            hand_line_thickness = max(15, int(face_size * 0.07))  # 7% of face width, minimum 15px
        
        # Create landmarks as numpy array for all hands
        for hand_landmark in hand_landmarks:
            hand_points = []
            for landmark in hand_landmark.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                hand_points.append((x, y))
                
                # Draw a circle around each landmark to make the hand mask thicker
                cv2.circle(hand_mask, (x, y), hand_point_radius, 255, -1)
                
            # Connect landmarks with lines to fill gaps
            if hand_points:
                for i in range(len(hand_points)-1):
                    cv2.line(hand_mask, hand_points[i], hand_points[i+1], 255, hand_line_thickness)
                    
        # Dilate the mask to ensure coverage
        kernel_size = max(15, int(hand_point_radius * 0.8))
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        hand_mask = cv2.dilate(hand_mask, kernel, iterations=1)
                
        return hand_mask
    
    def detect_edges(self, frame, alert_mask, hand_mask, mouth_mask):
        """
        Detect edges in the frame and filter by alert region (excluding hands).
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Detect edges using Canny
        edges = cv2.Canny(blurred, self.EDGE_THRESHOLD_LOW, self.EDGE_THRESHOLD_HIGH)
        
        # Dilate edges to make them more visible
        kernel = np.ones((3, 3), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Create a mask of alert area not covered by hands
        alert_without_hands = cv2.bitwise_and(alert_mask, cv2.bitwise_not(hand_mask))
        
        # Get edges only in the alert region (excluding hands)
        alert_edges = cv2.bitwise_and(edges_dilated, alert_without_hands)
        
        # Edge intersection with mouth for visualization
        mouth_edge_intersection = cv2.bitwise_and(edges_dilated, cv2.bitwise_and(mouth_mask, cv2.bitwise_not(hand_mask)))
        
        # Detection channels for debugging
        detection_channels = {
            'edges': edges,
            'edges_dilated': edges_dilated,
            'alert_without_hands': alert_without_hands,
            'alert_edges': alert_edges,
            'mouth_edge_intersection': mouth_edge_intersection
        }
        
        return alert_edges, detection_channels
    
    def detect_object_in_mouth_by_edges(self, frame, mouth_mask, alert_mask, hand_mask):
        """Detect if foreign objects are in mouth area using edge detection."""
        if alert_mask is None or np.sum(alert_mask > 0) == 0:
            return False, None, 0, {}
        
        # Detect edges in alert region
        alert_edges, detection_channels = self.detect_edges(frame, alert_mask, hand_mask, mouth_mask)
        
        # Calculate coverage ratio
        alert_area = np.sum(alert_mask > 0)
        edge_pixels = np.sum(alert_edges > 0)
        
        coverage_ratio = 0
        if alert_area > 0:
            coverage_ratio = edge_pixels / alert_area
            self.coverage_history.append(coverage_ratio)
            
            # Average over recent frames for stability
            avg_coverage = sum(self.coverage_history) / len(self.coverage_history)
            
            if avg_coverage > self.OVERLAP_THRESHOLD:
                return True, alert_edges, avg_coverage, detection_channels
        
        return False, alert_edges, coverage_ratio, detection_channels
    
    def alert_user(self):
        """Play alert sound when object in mouth is detected."""
        current_time = time.time()
        
        if current_time - self.last_alert_time > self.ALERT_COOLDOWN:
            print("ALERT: Object in mouth detected!")
            # Play beep sound (Windows)
            winsound.Beep(1000, 500)
            self.last_alert_time = current_time
    
    def visualize_results(self, frame, mouth_points, mouth_center, mouth_mask, alert_mask, 
                         hand_mask, edge_mask, object_detected, face_size):
        """Create visualization for main display."""
        display_image = frame.copy()
        
        # Visualize alert region (larger than mouth)
        if alert_mask is not None:
            # Draw alert mask as purple overlay with low opacity
            alert_overlay = display_image.copy()
            purple_alert = np.zeros_like(display_image)
            purple_alert[alert_mask > 0] = [128, 0, 128]  # Purple for alert region
            cv2.addWeighted(alert_overlay, 0.9, purple_alert, 0.1, 0, display_image)
        
        # Visualize mouth region
        if mouth_mask is not None:
            # Draw mouth mask as blue overlay
            mouth_overlay = display_image.copy()
            blue_mouth = np.zeros_like(display_image)
            blue_mouth[mouth_mask > 0] = [255, 0, 0]  # Blue for mouth region
            cv2.addWeighted(mouth_overlay, 0.7, blue_mouth, 0.3, 0, display_image)
            
            # Draw mouth outline
            if mouth_points:
                for i in range(len(mouth_points)):
                    cv2.line(display_image, mouth_points[i], mouth_points[(i+1) % len(mouth_points)], 
                            (0, 0, 255), 2)
            
            # Draw mouth center
            if mouth_center:
                cv2.circle(display_image, (int(mouth_center[0]), int(mouth_center[1])), 
                        5, (255, 0, 255), -1)
        
        # Visualize hand mask
        if hand_mask is not None and np.sum(hand_mask > 0) > 0:
            # Draw hand mask as green overlay
            hand_overlay = display_image.copy()
            green_hand = np.zeros_like(display_image)
            green_hand[hand_mask > 0] = [0, 255, 0]  # Green for hand region
            cv2.addWeighted(hand_overlay, 0.7, green_hand, 0.3, 0, display_image)
        
        # Visualize edge mask if detected
        if edge_mask is not None and np.sum(edge_mask > 0) > 0:
            # Draw edges as yellow overlay
            edge_overlay = display_image.copy()
            yellow_edges = np.zeros_like(display_image)
            yellow_edges[edge_mask > 0] = [0, 255, 255]  # Yellow for edges
            cv2.addWeighted(edge_overlay, 0.7, yellow_edges, 0.3, 0, display_image)
        
        # Draw warning if object detected for enough time
        if object_detected:
            # Draw red warning border
            cv2.rectangle(display_image, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 10)
            
            # Show warning text
            cv2.putText(display_image, "WARNING: Object in mouth detected!", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display detection status
        status_y = 60
        
        # Show coverage ratio
        avg_coverage = sum(self.coverage_history) / len(self.coverage_history) if self.coverage_history else 0
        cv2.putText(display_image, f"Edge coverage: {avg_coverage:.3f} (Threshold: {self.OVERLAP_THRESHOLD:.3f})", 
                  (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        status_y += 30
        
        # Show face size
        cv2.putText(display_image, f"Face size: {face_size:.1f} px, Alert factor: {self.ALERT_MARGIN_FACTOR:.1f}x", 
                  (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        status_y += 30
        
        # Show pixel counts
        if alert_mask is not None and edge_mask is not None:
            alert_pixels = np.sum(alert_mask > 0)
            edge_pixels = np.sum(edge_mask > 0)
            cv2.putText(display_image, f"Alert area: {alert_pixels}, Edge pixels: {edge_pixels}", 
                      (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            status_y += 30
        
        if self.object_detected_active:
            time_active = time.time() - self.object_detected_start_time
            cv2.putText(display_image, f"Object detected for: {time_active:.1f}s / {self.TIME_THRESHOLD:.1f}s", 
                      (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            status_y += 30
        
        # Show edge thresholds
        cv2.putText(display_image, f"Edge thresholds: {self.EDGE_THRESHOLD_LOW}/{self.EDGE_THRESHOLD_HIGH}", 
                  (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return display_image
    
    def create_debug_windows(self, detection_channels, frame_size):
        """Create debug visualization windows for all detection channels."""
        debug_windows = {}
        
        # Calculate a scale factor to make debug windows smaller
        scale_factor = 0.4
        debug_size = (int(frame_size[1] * scale_factor), int(frame_size[0] * scale_factor))
        
        for name, mask in detection_channels.items():
            # Convert grayscale mask to BGR for visualization
            if len(mask.shape) == 2:  # If grayscale
                mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            else:
                mask_vis = mask.copy()
            
            # Resize for better display
            mask_vis = cv2.resize(mask_vis, (debug_size[0], debug_size[1]))
            
            # Add label
            cv2.putText(mask_vis, name, (10, 20), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            debug_windows[name] = mask_vis
        
        return debug_windows
    
    def process_frame(self, frame):
        """Process a single frame."""
        if frame is None:
            return None, False, {}
        
        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        face_results = self.face_mesh.process(rgb_frame)
        hand_results = self.hands.process(rgb_frame)
        
        # Initialize variables
        mouth_points = []
        mouth_center = None
        mouth_mask = None
        alert_mask = None
        mouth_roi = None
        hand_mask = None
        object_in_mouth = False
        edge_mask = None
        detection_channels = {}
        face_size = 0
        
        # Process face landmarks
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                # Detect mouth
                mouth_points, mouth_center, mouth_mask, alert_mask, mouth_roi, face_size = self.detect_mouth(
                    face_landmarks, frame.shape)
        
        # Create hand mask
        if hand_results.multi_hand_landmarks:
            hand_mask = self.create_hand_mask(hand_results.multi_hand_landmarks, frame.shape, face_size)
        else:
            hand_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        
        # Detect object in mouth
        object_detected = False
        if mouth_mask is not None and alert_mask is not None and hand_mask is not None:
            object_in_mouth, edge_mask, coverage_ratio, detection_channels = self.detect_object_in_mouth_by_edges(
                frame, mouth_mask, alert_mask, hand_mask)
            
            # Time-based detection to avoid false positives
            current_time = time.time()
            if object_in_mouth:
                if not self.object_detected_active:
                    self.object_detected_active = True
                    self.object_detected_start_time = current_time
                elif current_time - self.object_detected_start_time > self.TIME_THRESHOLD:
                    # Alert if the behavior persists for TIME_THRESHOLD seconds
                    self.alert_user()
                    object_detected = True
            else:
                self.object_detected_active = False
        
        # Draw MediaPipe landmarks
        display_image = frame.copy()
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    display_image,
                    face_landmarks,
                    self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
                )
        
        if hand_results.multi_hand_landmarks:
            for hand_landmark in hand_results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    display_image,
                    hand_landmark,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        
        # Visualize results
        visual_output = self.visualize_results(
            display_image, mouth_points, mouth_center, mouth_mask, alert_mask,
            hand_mask, edge_mask, object_detected, face_size
        )
        
        # Create debug visualizations
        if self.debug_mode:
            # Add masks to detection channels if not already there
            if 'hand_mask' not in detection_channels and hand_mask is not None:
                detection_channels['hand_mask'] = hand_mask
            if 'mouth_mask' not in detection_channels and mouth_mask is not None:
                detection_channels['mouth_mask'] = mouth_mask
            if 'alert_mask' not in detection_channels and alert_mask is not None:
                detection_channels['alert_mask'] = alert_mask
            
            debug_windows = self.create_debug_windows(detection_channels, frame.shape)
        else:
            debug_windows = {}
        
        return visual_output, object_detected, debug_windows
    
    def run(self):
        """Run the detector on the webcam feed."""
        if not self.open_camera():
            return
        
        # Create main window
        cv2.namedWindow('Edge-Based Mouth Detector', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Edge-Based Mouth Detector', 1280, 720)
        
        # Create debug window grid
        if self.debug_mode:
            cv2.namedWindow('Detection Masks', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Detection Masks', 1280, 720)
        
        print("Controls:")
        print("  ESC: Exit")
        print("  +/-: Increase/decrease sensitivity (threshold)")
        print("  E/Q: Increase/decrease edge detection (low threshold)")
        print("  R/F: Increase/decrease edge detection (high threshold)")
        print("  A/Z: Increase/decrease alert area margin")
        print("  D: Toggle debug mode")
        
        while self.cap.isOpened():
            # Read frame
            success, frame = self.cap.read()
            if not success:
                print("Failed to capture image from camera.")
                break
            
            # Process frame
            display_image, _, debug_windows = self.process_frame(frame)
            
            # Display the main image
            cv2.imshow('Edge-Based Mouth Detector', display_image)
            
            # Display debug windows if enabled
            if self.debug_mode and debug_windows:
                # Create a grid of debug windows
                grid_cols = 3
                grid_rows = (len(debug_windows) + grid_cols - 1) // grid_cols
                
                # Get the size of the first debug window to determine grid dimensions
                first_window = next(iter(debug_windows.values()))
                debug_h, debug_w = first_window.shape[:2]
                
                # Create a blank canvas for the grid
                grid_h = debug_h * grid_rows
                grid_w = debug_w * grid_cols
                grid_canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
                
                # Place each debug window in the grid
                i = 0
                for name, window in debug_windows.items():
                    row = i // grid_cols
                    col = i % grid_cols
                    
                    y_start = row * debug_h
                    y_end = y_start + debug_h
                    x_start = col * debug_w
                    x_end = x_start + debug_w
                    
                    # Make sure the window fits in the grid dimensions
                    h, w = window.shape[:2]
                    h, w = min(h, debug_h), min(w, debug_w)
                    
                    grid_canvas[y_start:y_start+h, x_start:x_start+w] = window[:h, :w]
                    i += 1
                
                cv2.imshow('Detection Masks', grid_canvas)
            
            # Check for keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('+') or key == ord('='):
                self.OVERLAP_THRESHOLD = max(0.01, self.OVERLAP_THRESHOLD - 0.01)
                print(f"Sensitivity increased. Threshold: {self.OVERLAP_THRESHOLD:.3f}")
            elif key == ord('-') or key == ord('_'):
                self.OVERLAP_THRESHOLD = min(0.5, self.OVERLAP_THRESHOLD + 0.01)
                print(f"Sensitivity decreased. Threshold: {self.OVERLAP_THRESHOLD:.3f}")
            elif key == ord('e') or key == ord('E'):
                self.EDGE_THRESHOLD_LOW = min(100, self.EDGE_THRESHOLD_LOW + 5)
                print(f"Edge low threshold increased: {self.EDGE_THRESHOLD_LOW}")
            elif key == ord('q') or key == ord('Q'):
                self.EDGE_THRESHOLD_LOW = max(10, self.EDGE_THRESHOLD_LOW - 5)
                print(f"Edge low threshold decreased: {self.EDGE_THRESHOLD_LOW}")
            elif key == ord('r') or key == ord('R'):
                self.EDGE_THRESHOLD_HIGH = min(250, self.EDGE_THRESHOLD_HIGH + 5)
                print(f"Edge high threshold increased: {self.EDGE_THRESHOLD_HIGH}")
            elif key == ord('f') or key == ord('F'):
                self.EDGE_THRESHOLD_HIGH = max(self.EDGE_THRESHOLD_LOW + 10, self.EDGE_THRESHOLD_HIGH - 5)
                print(f"Edge high threshold decreased: {self.EDGE_THRESHOLD_HIGH}")
            elif key == ord('a') or key == ord('A'):
                self.ALERT_MARGIN_FACTOR = min(3.0, self.ALERT_MARGIN_FACTOR + 0.1)
                print(f"Alert margin increased: {self.ALERT_MARGIN_FACTOR:.1f}x")
            elif key == ord('z') or key == ord('Z'):
                self.ALERT_MARGIN_FACTOR = max(1.0, self.ALERT_MARGIN_FACTOR - 0.1)
                print(f"Alert margin decreased: {self.ALERT_MARGIN_FACTOR:.1f}x")
            elif key == ord('d') or key == ord('D'):
                self.debug_mode = not self.debug_mode
                print(f"Debug mode: {'ON' if self.debug_mode else 'OFF'}")
                if not self.debug_mode:
                    cv2.destroyWindow('Detection Masks')
        
        # Clean up
        self.cap.release()
        cv2.destroyAllWindows()
    
    def close(self):
        """Close resources."""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()

def main():
    """Main function."""
    detector = EdgeMouthDetector()
    try:
        detector.run()
    finally:
        detector.close()

if __name__ == "__main__":
    main()