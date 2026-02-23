"""
Test script to verify the helmet detection fix
"""

import cv2
import numpy as np
import time
import argparse
import onnxruntime as ort
import os

# Configuration with fixed threshold
MODEL_PATH = "helmet_detector_dynamic.onnx"
CONF_THRESHOLD = 0.25  # Fixed threshold based on diagnostic
IOU_THRESHOLD = 0.45
INPUT_SIZE = 256

# Class names
CLASS_NAMES = {0: 'helmet', 1: 'nohelmet'}

# Colors (BGR format)
COLORS = {
    'helmet': (0, 255, 0),      # Green
    'nohelmet': (0, 0, 255)     # Red
}

class HelmetDetectorFixed:
    """Fixed helmet detector with proper confidence threshold"""
    
    def __init__(self, model_path, conf_threshold=CONF_THRESHOLD, input_size=INPUT_SIZE):
        print("📦 Loading ONNX model (Fixed Version)...")
        self.input_size = input_size
        
        # Check if model exists
        if not os.path.exists(model_path):
            print(f"❌ Error: Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Setup session options
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Load model
        try:
            self.session = ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])
        except Exception as e:
            print(f"❌ Error: Failed to load model: {e}")
            raise RuntimeError(f"Failed to load model: {e}")
        
        self.conf_threshold = conf_threshold
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        print(f"✅ Model loaded: {model_path}")
        print(f"⚡ Fixed Configuration: INPUT_SIZE={input_size}, CONF={conf_threshold}")
        print(f"🎯 This threshold is optimized for models with low confidence scores")
        print(f"🚀 Ready for detection!\n")
    
    def preprocess(self, image):
        """Preprocess image for model input"""
        img = cv2.resize(image, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        img = img.astype(np.float32) / 255.0  # Normalize
        return img
    
    def postprocess(self, outputs, orig_shape):
        """Post-process model outputs"""
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
        
        mask = confidences > self.conf_threshold
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
    
    def detect(self, image):
        """Run detection on image"""
        # Preprocess
        input_data = self.preprocess(image)
        
        # Inference
        outputs = self.session.run(self.output_names, {self.input_name: input_data})
        
        # Postprocess
        detections = self.postprocess(outputs, image.shape)
        
        return detections

def test_detection():
    """Test helmet detection with fixed threshold"""
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model file not found: {MODEL_PATH}")
        return
    
    print(f"🎯 Initializing FIXED helmet detection system...")
    print(f"📁 Model file: {MODEL_PATH}")
    
    # Initialize detector
    try:
        detector = HelmetDetectorFixed(MODEL_PATH, CONF_THRESHOLD, INPUT_SIZE)
        print(f"🚀 Fixed detector initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize detector: {e}")
        return
    
    # Open video
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print(f"❌ Error: Cannot open camera")
        return
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Get video info
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"🎥 Camera Resolution: {width}x{height}")
    print(f"🎬 FPS: {fps}")
    print(f"🎯 Confidence Threshold: {CONF_THRESHOLD} (FIXED)")
    print(f"🚀 Starting FIXED detection... (Press 'q' to quit)\n")
    
    # Statistics
    frame_count = 0
    processed_count = 0
    total_helmets = 0
    total_nohelmets = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("\n✅ Video processing complete!")
                break
            
            frame_count += 1
            processed_count += 1
            
            # Run detection
            detections = detector.detect(frame)
            
            # Count detections
            helmets_in_frame = sum(1 for d in detections if d['class_name'] == 'helmet')
            nohelmets_in_frame = sum(1 for d in detections if d['class_name'] == 'nohelmet')
            
            total_helmets += helmets_in_frame
            total_nohelmets += nohelmets_in_frame
            
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
            
            # Calculate FPS
            elapsed_time = time.time() - start_time
            current_fps = processed_count / elapsed_time if elapsed_time > 0 else 0
            
            # Display info on frame
            info_y = 25
            cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.putText(frame, f"Frame: {frame_count}", (10, info_y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.putText(frame, f"H:{helmets_in_frame} | NH:{nohelmets_in_frame}",
                       (10, info_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.putText(frame, f"CONF: {CONF_THRESHOLD}", (10, info_y + 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Display
            cv2.imshow('Helmet Detection - FIXED VERSION', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n⏹️  Stopped by user")
                break
            
            # Progress update
            if processed_count % 30 == 0 and processed_count > 0:
                print(f"⏳ Frame {frame_count} | Processed: {processed_count} | "
                      f"FPS: {current_fps:.1f} | H:{total_helmets} NH:{total_nohelmets}")
    
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # Summary
        elapsed_time = time.time() - start_time
        print("\n" + "="*60)
        print("📊 FIXED Detection Summary")
        print("="*60)
        print(f"⏱️  Total Time: {elapsed_time:.2f} seconds")
        print(f"🎬 Total Frames: {frame_count}")
        print(f"⚡ Processed Frames: {processed_count}")
        print(f"📈 Processing FPS: {processed_count/elapsed_time:.2f}")
        print(f"✅ Total Helmets: {total_helmets}")
        print(f"❌ Total No Helmets: {total_nohelmets}")
        print(f"🎯 Confidence Threshold: {CONF_THRESHOLD} (FIXED)")
        print("="*60)

if __name__ == "__main__":
    test_detection()
