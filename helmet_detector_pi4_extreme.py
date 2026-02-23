"""
Helmet Detection for Raspberry Pi 4 - EXTREME TURBO MODE
เพิ่ม FPS ให้ได้มากกว่า 5+ FPS ด้วยการปรับแต่งขั้นสูงสุด
"""

import cv2
import numpy as np
import time
import argparse
import onnxruntime as ort
import gc
import os

# ========== EXTREME TURBO Configuration ==========
MODEL_PATH = "helmet_detector_dynamic.onnx"
CONF_THRESHOLD = 0.25  # ค่าที่เหมาะสมจากการทดสอบ
IOU_THRESHOLD = 0.45

# EXTREME Settings - ปรับเพื่อ FPS สูงสุดเกิน 5+
EXTREME_INPUT_SIZE = 160   # เล็กมากๆ
EXTREME_CAM_WIDTH = 128    # ความละเอียดต่ำสุด
EXTREME_CAM_HEIGHT = 96
EXTREME_SKIP_FRAMES = 12   # ข้ามเฟรมเยอะที่สุด
EXTREME_THREADS = 1        # Single thread เพื่อลด overhead

# Class names
CLASS_NAMES = {0: 'helmet', 1: 'nohelmet'}

# Colors (BGR format)
COLORS = {
    'helmet': (0, 255, 0),      # Green
    'nohelmet': (0, 0, 255)     # Red
}

class HelmetDetectorExtreme:
    """
    EXTREME TURBO helmet detector - มุ่งเป้า >5 FPS
    ใช้เทคนิคขั้นสูงสุดเพื่อความเร็ว
    """
    
    def __init__(self, model_path, conf_threshold=CONF_THRESHOLD, input_size=EXTREME_INPUT_SIZE):
        print("🔥 Loading ONNX model (EXTREME TURBO)...")
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        
        # Check model
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # EXTREME optimized session options
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = EXTREME_THREADS
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.enable_mem_pattern = True
        opts.enable_cpu_mem_arena = True
        opts.enable_profiling = False
        opts.log_severity_level = 3  # ปิด logging
        
        # Load model
        self.session = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"]
        )
        
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        # Pre-allocate buffers
        self._input_buffer = np.zeros((1, 3, input_size, input_size), dtype=np.float32)
        
        print(f"✅ EXTREME Model loaded")
        print(f"🔥 Input Size: {input_size}x{input_size}")
        print(f"🎯 Confidence: {conf_threshold}")
        print(f"🧵 Threads: {EXTREME_THREADS}")
        print(f"🚀 EXTREME TURBO Ready!")
    
    def preprocess(self, image):
        """Ultra-fast preprocessing"""
        # Resize ด้วย INTER_NEAREST (เร็วที่สุด)
        resized = cv2.resize(image, (self.input_size, self.input_size), 
                            interpolation=cv2.INTER_NEAREST)
        
        # Direct assignment to pre-allocated buffer
        self._input_buffer[0] = resized.transpose(2, 0, 1).astype(np.float32) / 255.0
        
        return self._input_buffer
    
    def postprocess(self, outputs, orig_shape):
        """Minimal postprocessing for speed"""
        detections = []
        output = outputs[0]
        
        # Fast shape handling
        if len(output.shape) == 3:
            output = output[0].T if output.shape[1] < output.shape[2] else output[0]
        else:
            output = output[0]
        
        # Vectorized filtering
        class_scores = output[:, 4:]
        confidences = np.max(class_scores, axis=1)
        
        mask = confidences > self.conf_threshold
        if not np.any(mask):
            return detections
        
        filtered = output[mask]
        filtered_conf = confidences[mask]
        class_ids = np.argmax(filtered[:, 4:], axis=1)
        
        # Fast coordinate conversion
        orig_h, orig_w = orig_shape[:2]
        scale_x = orig_w / self.input_size
        scale_y = orig_h / self.input_size
        
        x_c = filtered[:, 0]; y_c = filtered[:, 1]
        w = filtered[:, 2]; h = filtered[:, 3]
        
        x1 = np.clip(((x_c - w/2) * scale_x).astype(np.int32), 0, orig_w)
        y1 = np.clip(((y_c - h/2) * scale_y).astype(np.int32), 0, orig_h)
        x2 = np.clip(((x_c + w/2) * scale_x).astype(np.int32), 0, orig_w)
        y2 = np.clip(((y_c + h/2) * scale_y).astype(np.int32), 0, orig_h)
        
        # Build detections (minimal loop)
        for i in range(len(filtered)):
            cid = int(class_ids[i])
            if cid in CLASS_NAMES:
                detections.append({
                    'box': (int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])),
                    'confidence': float(filtered_conf[i]),
                    'class_id': cid,
                    'class_name': CLASS_NAMES[cid]
                })
        
        return detections
    
    def detect(self, image):
        """Ultra-fast detection"""
        input_data = self.preprocess(image)
        outputs = self.session.run(self.output_names, {self.input_name: input_data})
        return self.postprocess(outputs, image.shape)

def detect_video_extreme(source=0, output_path=None, show_display=True,
                        skip_frames=EXTREME_SKIP_FRAMES,
                        cam_width=EXTREME_CAM_WIDTH, cam_height=EXTREME_CAM_HEIGHT,
                        input_size=EXTREME_INPUT_SIZE,
                        headless=False):
    """
    EXTREME TURBO MODE - มุ่งเป้า >5 FPS
    """
    
    print("🔥 INITIALIZING EXTREME TURBO MODE...")
    print(f"📹 Camera: {cam_width}x{cam_height}")
    print(f"🔥 Input Size: {input_size}x{input_size}")
    print(f"⚡ Skip Frames: {skip_frames}")
    print(f"🎯 TARGET: 5+ FPS")
    
    # Initialize detector
    detector = HelmetDetectorExtreme(MODEL_PATH, CONF_THRESHOLD, input_size)
    
    # Open camera with EXTREME settings
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    # EXTREME camera settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    
    # Get actual settings
    actual_fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"🎥 Actual Camera: {actual_width}x{actual_height} @ {actual_fps}fps")
    
    # Video writer (optional)
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, actual_fps, (actual_width, actual_height))
        print(f"💾 Recording to: {output_path}")
    
    # Statistics
    frame_count = 0
    processed_count = 0
    total_helmets = 0
    total_nohelmets = 0
    start_time = time.time()
    last_detections = []
    last_process_time = 0
    
    show_window = show_display and not headless
    print("🔥 EXTREME TURBO STARTED! (Press 'q' to quit)")
    print("=" * 50)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            current_time = time.time()
            
            # Process only every Nth frame
            if frame_count % skip_frames == 0:
                process_start = time.time()
                
                # Run detection
                detections = detector.detect(frame)
                last_detections = detections
                last_process_time = time.time() - process_start
                
                processed_count += 1
                
                # Count (simplified)
                helmets = sum(1 for d in detections if d['class_name'] == 'helmet')
                nohelmets = sum(1 for d in detections if d['class_name'] == 'nohelmet')
                total_helmets += helmets
                total_nohelmets += nohelmets
            
            # Minimal drawing for speed
            for det in last_detections:
                x1, y1, x2, y2 = det['box']
                color = COLORS.get(det['class_name'], (255, 255, 255))
                
                # Draw box (1 pixel for speed)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                
                # Minimal label
                label = f"{det['class_name'][0]}"  # H or N only
                cv2.putText(frame, label, (x1, y1 - 3),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            # Calculate FPS
            elapsed = current_time - start_time
            display_fps = frame_count / elapsed if elapsed > 0 else 0
            process_fps = processed_count / elapsed if elapsed > 0 else 0
            
            # Minimal info display
            h_now = sum(1 for d in last_detections if d['class_name'] == 'helmet')
            nh_now = sum(1 for d in last_detections if d['class_name'] == 'nohelmet')
            
            # Single line info for speed
            info_text = f"FPS:{display_fps:.1f} P:{process_fps:.1f} H:{h_now} NH:{nh_now}"
            cv2.putText(frame, info_text, (2, 12), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
            
            # Save frame
            if writer:
                writer.write(frame)
            
            # Display
            if show_window:
                cv2.imshow('Helmet Detection - EXTREME', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            
            # Progress report (every 50 processed frames)
            if processed_count > 0 and processed_count % 50 == 0:
                print(f"🔥 Frame {frame_count} | Display FPS: {display_fps:.1f} | "
                      f"Process FPS: {process_fps:.1f} | Time: {last_process_time*1000:.1f}ms | "
                      f"H:{total_helmets} NH:{total_nohelmets}")
            
            # GC every 200 frames
            if frame_count % 200 == 0:
                gc.collect()
    
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted")
    
    finally:
        cap.release()
        if writer:
            writer.release()
        if show_window:
            cv2.destroyAllWindows()
        gc.collect()
        
        # Final summary
        elapsed = time.time() - start_time
        print("\n" + "=" * 60)
        print("📊 EXTREME TURBO - Performance Summary")
        print("=" * 60)
        print(f"⏱️  Total Time: {elapsed:.2f} seconds")
        print(f"🎬 Total Frames: {frame_count}")
        print(f"⚡ Processed Frames: {processed_count}")
        print(f"📈 Display FPS: {frame_count/elapsed:.2f}")
        print(f"🔥 Processing FPS: {processed_count/elapsed:.2f}")
        print(f"✅ Total Helmets: {total_helmets}")
        print(f"❌ Total No Helmets: {total_nohelmets}")
        print(f"⚡ Skip Frames: {skip_frames}")
        print(f"🔥 Input Size: {input_size}x{input_size}")
        if output_path:
            print(f"💾 Output: {output_path}")
        print("=" * 60)
        
        # Performance analysis
        if processed_count/elapsed >= 5:
            print("🎉🎉 SUCCESS: ACHIEVED 5+ FPS! 🎉🎉")
        else:
            print("⚠️  Still below 5 FPS - try even more aggressive settings")

def main():
    parser = argparse.ArgumentParser(
        description='Helmet Detection - Pi4 EXTREME TURBO (>5 FPS)',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
EXTREME TURBO Examples:
  # Default extreme settings (5+ FPS target)
  python3 helmet_detector_pi4_extreme.py
  
  # Maximum FPS (skip more frames)
  python3 helmet_detector_pi4_extreme.py --skip-frames 15
  
  # Slightly better quality
  python3 helmet_detector_pi4_extreme.py --input-size 192 --skip-frames 10
  
  # Save video
  python3 helmet_detector_pi4_extreme.py --output extreme_result.mp4
        """
    )
    
    parser.add_argument('--source', type=str, default='0',
                        help='Video source (default: 0 for webcam)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output video path (optional)')
    parser.add_argument('--skip-frames', type=int, default=EXTREME_SKIP_FRAMES,
                        help=f'Skip frames (default: {EXTREME_SKIP_FRAMES})')
    parser.add_argument('--cam-width', type=int, default=EXTREME_CAM_WIDTH,
                        help=f'Camera width (default: {EXTREME_CAM_WIDTH})')
    parser.add_argument('--cam-height', type=int, default=EXTREME_CAM_HEIGHT,
                        help=f'Camera height (default: {EXTREME_CAM_HEIGHT})')
    parser.add_argument('--input-size', type=int, default=EXTREME_INPUT_SIZE,
                        choices=[128, 160, 192, 224],
                        help=f'Model input size (default: {EXTREME_INPUT_SIZE})')
    parser.add_argument('--headless', action='store_true',
                        help='Run without display')
    parser.add_argument('--no-display', action='store_true',
                        help='Same as --headless')
    
    args = parser.parse_args()
    
    # Convert source
    source = args.source
    if source.isdigit():
        source = int(source)
    
    # Run extreme detection
    detect_video_extreme(
        source=source,
        output_path=args.output,
        show_display=True,
        skip_frames=args.skip_frames,
        cam_width=args.cam_width,
        cam_height=args.cam_height,
        input_size=args.input_size,
        headless=args.headless or args.no_display
    )

if __name__ == "__main__":
    main()
