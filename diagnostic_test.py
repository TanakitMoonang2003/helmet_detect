"""
Diagnostic script for helmet detection model
Tests different confidence thresholds and shows detection results
"""

import cv2
import numpy as np
import time
import onnxruntime as ort
import os

# Configuration
MODEL_PATH = "helmet_detector_dynamic.onnx"
INPUT_SIZE = 256
CLASS_NAMES = {0: 'helmet', 1: 'nohelmet'}
COLORS = {'helmet': (0, 255, 0), 'nohelmet': (0, 0, 255)}

class HelmetDetectorDiagnostic:
    def __init__(self, model_path, input_size=INPUT_SIZE):
        print("📦 Loading model for diagnostic...")
        
        # Load model
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.input_size = input_size
        
        print(f"✅ Model loaded: {model_path}")
        print(f"📊 Input shape: {self.session.get_inputs()[0].shape}")
        print(f"📊 Output shape: {self.session.get_outputs()[0].shape}")
    
    def preprocess(self, image):
        """Preprocess image for model input"""
        img = cv2.resize(image, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        img = img.astype(np.float32) / 255.0  # Normalize
        return img
    
    def detect_with_threshold(self, image, conf_threshold):
        """Run detection with specific confidence threshold"""
        # Preprocess
        input_data = self.preprocess(image)
        
        # Inference
        outputs = self.session.run(self.output_names, {self.input_name: input_data})
        
        # Postprocess with specific threshold
        detections = self.postprocess(outputs, image.shape, conf_threshold)
        
        return detections
    
    def postprocess(self, outputs, orig_shape, conf_threshold):
        """Post-process model outputs with specific confidence threshold"""
        detections = []
        output = outputs[0]
        
        # Handle output shape
        if len(output.shape) == 3:
            if output.shape[1] < output.shape[2]:
                output = output[0].T  # Transpose [6, 8400] -> [8400, 6]
            else:
                output = output[0]
        else:
            output = output[0]
            
        # Filter by confidence threshold
        class_scores = output[:, 4:]
        confidences = np.max(class_scores, axis=1)
        
        mask = confidences > conf_threshold
        filtered_output = output[mask]
        filtered_conf = confidences[mask]
        
        if len(filtered_output) == 0:
            return detections
            
        # Get class IDs
        class_ids = np.argmax(filtered_output[:, 4:], axis=1)
        
        # Convert coordinates
        orig_h, orig_w = orig_shape[:2]
        x_centers = filtered_output[:, 0]
        y_centers = filtered_output[:, 1]
        widths = filtered_output[:, 2]
        heights = filtered_output[:, 3]
        
        x1 = ((x_centers - widths / 2) * orig_w / self.input_size).astype(int)
        y1 = ((y_centers - heights / 2) * orig_h / self.input_size).astype(int)
        x2 = ((x_centers + widths / 2) * orig_w / self.input_size).astype(int)
        y2 = ((y_centers + heights / 2) * orig_h / self.input_size).astype(int)
        
        # Clip to image boundaries
        x1 = np.clip(x1, 0, orig_w)
        y1 = np.clip(y1, 0, orig_h)
        x2 = np.clip(x2, 0, orig_w)
        y2 = np.clip(y2, 0, orig_h)
        
        # Store results
        for i in range(len(filtered_output)):
            class_id = int(class_ids[i])
            if class_id in CLASS_NAMES:
                detections.append({
                    'box': (int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])),
                    'confidence': float(filtered_conf[i]),
                    'class_id': class_id,
                    'class_name': CLASS_NAMES[class_id]
                })
        
        return detections

def test_thresholds():
    """Test different confidence thresholds with webcam"""
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found: {MODEL_PATH}")
        return
    
    # Initialize detector
    detector = HelmetDetectorDiagnostic(MODEL_PATH)
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Test different thresholds
    thresholds = [0.3, 0.5, 0.7, 0.8, 0.9]
    current_threshold_idx = 0
    
    print("🔍 Diagnostic Mode - Press 't' to change threshold, 'q' to quit")
    print(f"📊 Testing thresholds: {thresholds}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get current threshold
        current_threshold = thresholds[current_threshold_idx]
        
        # Run detection
        detections = detector.detect_with_threshold(frame, current_threshold)
        
        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = det['box']
            color = COLORS.get(det['class_name'], (255, 255, 255))
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{det['class_name']}: {det['confidence']:.3f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                        (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Count detections
        helmets = sum(1 for d in detections if d['class_name'] == 'helmet')
        nohelmets = sum(1 for d in detections if d['class_name'] == 'nohelmet')
        
        # Display info
        info_text = [
            f"Threshold: {current_threshold:.2f}",
            f"Helmets: {helmets} | No Helmets: {nohelmets}",
            f"Total Detections: {len(detections)}",
            "Press 't' to change threshold, 'q' to quit"
        ]
        
        for i, text in enumerate(info_text):
            cv2.putText(frame, text, (10, 30 + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Show frame
        cv2.imshow('Helmet Detection Diagnostic', frame)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('t'):
            current_threshold_idx = (current_threshold_idx + 1) % len(thresholds)
            print(f"🔄 Changed threshold to: {thresholds[current_threshold_idx]:.2f}")
    
    cap.release()
    cv2.destroyAllWindows()

def analyze_model_output():
    """Analyze raw model output to understand the issue"""
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found: {MODEL_PATH}")
        return
    
    print("🔬 Analyzing raw model output...")
    
    detector = HelmetDetectorDiagnostic(MODEL_PATH)
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    ret, frame = cap.read()
    if not ret:
        print("❌ Cannot read frame")
        return
    
    # Preprocess
    input_data = detector.preprocess(frame)
    
    # Run inference
    outputs = detector.session.run(detector.output_names, {detector.input_name: input_data})
    output = outputs[0]
    
    print(f"📊 Raw output shape: {output.shape}")
    print(f"📊 Raw output type: {output.dtype}")
    print(f"📊 Raw output min: {np.min(output):.6f}")
    print(f"📊 Raw output max: {np.max(output):.6f}")
    print(f"📊 Raw output mean: {np.mean(output):.6f}")
    
    # Analyze confidence scores
    if len(output.shape) == 3:
        if output.shape[1] < output.shape[2]:
            output = output[0].T
        else:
            output = output[0]
    else:
        output = output[0]
    
    class_scores = output[:, 4:]
    confidences = np.max(class_scores, axis=1)
    
    print(f"📊 Confidence scores shape: {confidences.shape}")
    print(f"📊 Confidence min: {np.min(confidences):.6f}")
    print(f"📊 Confidence max: {np.max(confidences):.6f}")
    print(f"📊 Confidence mean: {np.mean(confidences):.6f}")
    print(f"📊 Confidence std: {np.std(confidences):.6f}")
    
    # Show distribution
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    for thresh in thresholds:
        count = np.sum(confidences > thresh)
        print(f"📊 Detections above {thresh:.1f}: {count}")
    
    # Show top 10 detections
    top_indices = np.argsort(confidences)[-10:][::-1]
    print("\n🎯 Top 10 detections:")
    for i, idx in enumerate(top_indices):
        class_id = np.argmax(class_scores[idx])
        class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")
        print(f"  {i+1}. {class_name}: {confidences[idx]:.6f}")
    
    cap.release()

if __name__ == "__main__":
    print("🔍 Helmet Detection Diagnostic Tool")
    print("=" * 50)
    print("1. Test different confidence thresholds")
    print("2. Analyze raw model output")
    print("=" * 50)
    
    choice = input("Choose option (1 or 2): ").strip()
    
    if choice == "1":
        test_thresholds()
    elif choice == "2":
        analyze_model_output()
    else:
        print("❌ Invalid choice")
