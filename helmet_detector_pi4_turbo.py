"""
Helmet Detection for Raspberry Pi 4 - TURBO MODE
เพิ่ม FPS ให้ได้มากกว่า 5+ FPS ด้วยการปรับแต่งขั้นสูง
"""

import cv2
import numpy as np
import time
import argparse
import onnxruntime as ort
import gc
import os
import threading
from queue import Queue

# ========== TURBO Configuration ==========
MODEL_PATH = "helmet_detector_dynamic.onnx"
CONF_THRESHOLD = 0.25  # ค่าที่เหมาะสมจากการทดสอบ
IOU_THRESHOLD = 0.45

# TURBO Settings - ปรับเพื่อ FPS สูงสุด
TURBO_INPUT_SIZE = 192  # เล็กที่สุดที่ยังใช้ได้
TURBO_CAM_WIDTH = 160   # ความละเอียดกล้องต่ำสุด
TURBO_CAM_HEIGHT = 120
TURBO_SKIP_FRAMES = 8   # ข้ามเฟรมเยอะๆ
TURBO_THREADS = 1       # ลด thread contention

# Class names
CLASS_NAMES = {0: 'helmet', 1: 'nohelmet'}

# Colors (BGR format)
COLORS = {
    'helmet': (0, 255, 0),      # Green
    'nohelmet': (0, 0, 255)     # Red
}

class HelmetDetectorTurbo:
    """
    Ultra-fast helmet detector for maximum FPS on Pi 4
    ใช้เทคนิคพิเศษหลายอย่างเพื่อเพิ่มความเร็ว
    """
    
    def __init__(self, model_path, conf_threshold=CONF_THRESHOLD, input_size=TURBO_INPUT_SIZE):
        print("🚀 Loading ONNX model (TURBO MODE)...")
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        
        # Check model
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Ultra-optimized session options
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = TURBO_THREADS
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.enable_mem_pattern = True
        opts.enable_cpu_mem_arena = True
        opts.enable_profiling = False  # ปิด profiling เพื่อความเร็ว
        
        # Load model
        self.session = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"]
        )
        
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        # Pre-allocate buffers เพื่อลด memory allocation
        self._input_buffer = np.zeros((1, 3, input_size, input_size), dtype=np.float32)
        self._temp_frame = None
        
        print(f"✅ TURBO Model loaded")
        print(f"⚡ Input Size: {input_size}x{input_size}")
        print(f"🎯 Confidence: {conf_threshold}")
        print(f"🧵 Threads: {TURBO_THREADS}")
        print(f"🚀 TURBO MODE Ready!")
    
    def preprocess(self, image):
        """Ultra-fast preprocessing with pre-allocated buffer"""
        # Resize ด้วย INTER_NEAREST (เร็วที่สุด)
        resized = cv2.resize(image, (self.input_size, self.input_size), 
                            interpolation=cv2.INTER_NEAREST)
        
        # เขียนลง pre-allocated buffer
        np.copyto(self._input_buffer[0], 
                 resized.transpose(2, 0, 1).astype(np.float32) / 255.0)
        
        return self._input_buffer
    
    def postprocess(self, outputs, orig_shape):
        """Optimized postprocessing"""
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
        
        # Build detections
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

class FrameBuffer:
    """Buffer for smooth frame processing"""
    def __init__(self, max_size=3):
        self.buffer = Queue(maxsize=max_size)
        self.last_frame = None
    
    def add_frame(self, frame):
        try:
            self.buffer.put(frame, block=False)
        except:
            # Buffer full, drop oldest
            try:
                self.buffer.get(block=False)
                self.buffer.put(frame, block=False)
            except:
                pass
    
    def get_frame(self):
        try:
            frame = self.buffer.get(block=False)
            self.last_frame = frame
            return frame
        except:
            return self.last_frame

def detect_video_turbo(source=0, output_path=None, show_display=True,
                      skip_frames=TURBO_SKIP_FRAMES,
                      cam_width=TURBO_CAM_WIDTH, cam_height=TURBO_CAM_HEIGHT,
                      input_size=TURBO_INPUT_SIZE,
                      headless=False):
    """
    TURBO MODE helmet detection - สำหรับ FPS สูงสุด
    """
    
    print("🚀 INITIALIZING TURBO MODE...")
    print(f"📹 Camera: {cam_width}x{cam_height}")
    print(f"🔬 Input Size: {input_size}x{input_size}")
    print(f"⚡ Skip Frames: {skip_frames}")
    print(f"🎯 Target: 5+ FPS")
    
    # Initialize detector
    detector = HelmetDetectorTurbo(MODEL_PATH, CONF_THRESHOLD, input_size)
    
    # Open camera with TURBO settings
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    # Aggressive camera settings for max FPS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)   # Disable autofocus
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual exposure
    
    # Get actual camera settings
    actual_fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"🎥 Actual Camera: {actual_width}x{actual_height} @ {actual_fps}fps")
    
    # Frame buffer for smooth display
    frame_buffer = FrameBuffer()
    
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
    print("🚀 TURBO MODE STARTED! (Press 'q' to quit)")
    print("=" * 50)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            current_time = time.time()
            
            # Process only every Nth frame for TURBO performance
            if frame_count % skip_frames == 0:
                process_start = time.time()
                
                # Run detection
                detections = detector.detect(frame)
                last_detections = detections
                last_process_time = time.time() - process_start
                
                processed_count += 1
                
                # Count
                helmets = sum(1 for d in detections if d['class_name'] == 'helmet')
                nohelmets = sum(1 for d in detections if d['class_name'] == 'nohelmet')
                total_helmets += helmets
                total_nohelmets += nohelmets
            
            # Draw detections (use last detections for skipped frames)
            for det in last_detections:
                x1, y1, x2, y2 = det['box']
                color = COLORS.get(det['class_name'], (255, 255, 255))
                
                # Draw box (thinner for speed)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                
                # Draw label (smaller font for speed)
                label = f"{det['class_name'][:1]}:{det['confidence']:.2f}"
                cv2.putText(frame, label, (x1, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Calculate FPS
            elapsed = current_time - start_time
            display_fps = frame_count / elapsed if elapsed > 0 else 0
            process_fps = processed_count / elapsed if elapsed > 0 else 0
            
            # Minimal info display (for speed)
            h_now = sum(1 for d in last_detections if d['class_name'] == 'helmet')
            nh_now = sum(1 for d in last_detections if d['class_name'] == 'nohelmet')
            
            info_lines = [
                f"FPS:{display_fps:.1f}",
                f"P:{process_fps:.1f}",
                f"H:{h_now} NH:{nh_now}",
                f"S:{skip_frames}"
            ]
            
            for i, line in enumerate(info_lines):
                cv2.putText(frame, line, (5, 15 + i*12), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
            
            # Save frame
            if writer:
                writer.write(frame)
            
            # Display
            if show_window:
                cv2.imshow('Helmet Detection - TURBO', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            
            # Progress report (less frequent for speed)
            if processed_count > 0 and processed_count % 20 == 0:
                print(f"⚡ Frame {frame_count} | Display FPS: {display_fps:.1f} | "
                      f"Process FPS: {process_fps:.1f} | Process Time: {last_process_time*1000:.1f}ms | "
                      f"H:{total_helmets} NH:{total_nohelmets}")
            
            # Aggressive GC every 100 frames
            if frame_count % 100 == 0:
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
        print("📊 TURBO MODE - Performance Summary")
        print("=" * 60)
        print(f"⏱️  Total Time: {elapsed:.2f} seconds")
        print(f"🎬 Total Frames: {frame_count}")
        print(f"⚡ Processed Frames: {processed_count}")
        print(f"📈 Display FPS: {frame_count/elapsed:.2f}")
        print(f"🔬 Processing FPS: {processed_count/elapsed:.2f}")
        print(f"✅ Total Helmets: {total_helmets}")
        print(f"❌ Total No Helmets: {total_nohelmets}")
        print(f"🎯 Skip Frames: {skip_frames}")
        print(f"🔬 Input Size: {input_size}x{input_size}")
        if output_path:
            print(f"💾 Output: {output_path}")
        print("=" * 60)
        
        # Performance analysis
        if processed_count/elapsed >= 5:
            print("🎉 SUCCESS: Achieved 5+ FPS!")
        else:
            print("⚠️  Note: FPS below 5, try reducing input size or increasing skip frames")

def main():
    parser = argparse.ArgumentParser(
        description='Helmet Detection - Pi4 TURBO MODE (>5 FPS)',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
TURBO MODE Examples:
  # Default turbo settings (5+ FPS target)
  python3 helmet_detector_pi4_turbo.py
  
  # Ultra turbo (maximum FPS)
  python3 helmet_detector_pi4_turbo.py --skip-frames 10 --input-size 160
  
  # Balanced turbo
  python3 helmet_detector_pi4_turbo.py --skip-frames 5 --input-size 224
  
  # Save video
  python3 helmet_detector_pi4_turbo.py --output turbo_result.mp4
        """
    )
    
    parser.add_argument('--source', type=str, default='0',
                        help='Video source (default: 0 for webcam)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output video path (optional)')
    parser.add_argument('--skip-frames', type=int, default=TURBO_SKIP_FRAMES,
                        help=f'Skip frames (default: {TURBO_SKIP_FRAMES})\n'
                             f'  Higher = more FPS, less accuracy')
    parser.add_argument('--cam-width', type=int, default=TURBO_CAM_WIDTH,
                        help=f'Camera width (default: {TURBO_CAM_WIDTH})')
    parser.add_argument('--cam-height', type=int, default=TURBO_CAM_HEIGHT,
                        help=f'Camera height (default: {TURBO_CAM_HEIGHT})')
    parser.add_argument('--input-size', type=int, default=TURBO_INPUT_SIZE,
                        choices=[128, 160, 192, 224, 256],
                        help=f'Model input size (default: {TURBO_INPUT_SIZE})\n'
                             f'  Smaller = faster, less accurate')
    parser.add_argument('--headless', action='store_true',
                        help='Run without display (headless mode)')
    parser.add_argument('--no-display', action='store_true',
                        help='Same as --headless')
    
    args = parser.parse_args()
    
    # Convert source
    source = args.source
    if source.isdigit():
        source = int(source)
    
    # Run turbo detection
    detect_video_turbo(
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
