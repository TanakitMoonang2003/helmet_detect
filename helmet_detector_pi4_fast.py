"""
Helmet Detection for Raspberry Pi 4 - High Performance Version
Optimized for maximum FPS with acceptable accuracy
"""

import cv2
import numpy as np
import time
import argparse
import onnxruntime as ort
import os

# ========== Configuration ==========
# ใช้โมเดลแบบ Dynamic เพื่อรองรับ input หลายขนาด (320, 416, 640)
MODEL_PATH = "helmet_detector_dynamic.onnx"
CONF_THRESHOLD = 0.65  # เพิ่มจาก 0.5 เพื่อลดการตรวจจับที่ไม่จำเป็น
IOU_THRESHOLD = 0.45
# ลด INPUT_SIZE ลงเหลือ 320 เพื่อให้รันบน CPU ของ Raspberry Pi ได้ลื่นขึ้นมาก
# (สามารถ override ได้ด้วย --input-size argument)
INPUT_SIZE = 320

# Class names
CLASS_NAMES = {0: 'helmet', 1: 'nohelmet'}

# Colors (BGR format)
COLORS = {
    'helmet': (0, 255, 0),      # Green
    'nohelmet': (0, 0, 255)     # Red
}

class HelmetDetector:
    """Helmet detector optimized for Raspberry Pi 4"""
    
    def __init__(self, model_path, conf_threshold=0.5, input_size=INPUT_SIZE):
        """Initialize the detector"""
        print("📦 Loading ONNX model...")
        self.input_size = input_size
        
        # ตรวจสอบว่าไฟล์โมเดลมีอยู่จริงหรือไม่
        if not os.path.exists(model_path):
            print(f"❌ Error: Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # ตั้งค่า Session Options เพื่อเพิ่มความเร็วบน Pi 4 (CPU only)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4  # ใช้ครบ 4 Core ของ Pi 4
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # บังคับใช้ CPUExecutionProvider (ใน Pi ส่วนใหญ่มี provider ตัวนี้ตัวเดียว)
        try:
            self.session = ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])
        except Exception as e:
            print(f"❌ Error: Failed to load model: {e}")
            raise RuntimeError(f"Failed to load model: {e}")
        
        self.conf_threshold = conf_threshold
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        print(f"✅ Model loaded: {model_path}")
        print(f"⚡ Performance Mode: INPUT_SIZE={input_size}, CONF={conf_threshold}")
        print(f"🧵 Threads: 4 (Optimized for Pi 4)")
        
        # ทดสอบว่าโมเดลทำงานได้จริงหรือไม่
        self._test_model()
        print(f"🎯 Model test completed - Ready for detection!\n")
    
    def _test_model(self):
        """ทดสอบว่าโมเดลสามารถทำงานได้จริง"""
        print("🔍 Testing model functionality...")
        
        try:
            # สร้าง dummy input สำหรับทดสอบ
            dummy_input = np.random.rand(1, 3, self.input_size, self.input_size).astype(np.float32)
            
            # ทดสอบ inference
            start_time = time.time()
            outputs = self.session.run(self.output_names, {self.input_name: dummy_input})
            inference_time = time.time() - start_time
            
            # ตรวจสอบ output
            if len(outputs) > 0:
                output_shape = outputs[0].shape
                print(f"   📊 Output shape: {output_shape}")
                print(f"   ⏱️  Test inference time: {inference_time*1000:.2f}ms")
                
                # ทดสอบ postprocess ด้วย dummy output
                dummy_shape = (480, 640, 3)  # ขนาดภาพจริง
                test_detections = self.postprocess(outputs, dummy_shape)
                print(f"   🎯 Test detections: {len(test_detections)} objects found")
                
                if len(test_detections) > 0:
                    print(f"   📋 Sample detection: {test_detections[0]['class_name']} ({test_detections[0]['confidence']:.3f})")
                
                print("   ✅ Model test PASSED")
            else:
                print("   ❌ Model test FAILED: No output")
                raise RuntimeError("Model produced no output")
                
        except Exception as e:
            print(f"   ❌ Model test FAILED: {e}")
            raise RuntimeError(f"Model test failed: {e}")
    
    def preprocess(self, image):
        """Preprocess image for model input"""
        # Resize using INTER_LINEAR (เร็วกว่า AREA)
        img = cv2.resize(image, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        img = img.astype(np.float32) / 255.0  # Normalize
        return img
    
    def postprocess(self, outputs, orig_shape):
        """Post-process model outputs using NumPy Vectorization (FAST)"""
        detections = []
        output = outputs[0]
        
        # 1. จัดการ Shape ของ Output
        if len(output.shape) == 3:
            if output.shape[1] < output.shape[2]:
                output = output[0].T  # Transpose [6, 8400] -> [8400, 6]
            else:
                output = output[0]
        else:
            output = output[0]
            
        # 2. ใช้ NumPy กรองข้อมูลแทน Loop (Vectorized Filtering)
        # YOLOv8 format: [x, y, w, h, score_class0, score_class1, ...]
        class_scores = output[:, 4:]
        confidences = np.max(class_scores, axis=1)
        
        # กรองเฉพาะกล่องที่ confidence เกิน threshold
        mask = confidences > self.conf_threshold
        filtered_output = output[mask]
        filtered_conf = confidences[mask]
        
        if len(filtered_output) == 0:
            return detections
            
        # 3. หา Class ID ของแต่ละกล่อง
        class_ids = np.argmax(filtered_output[:, 4:], axis=1)
        
        # 4. แปลงพิกัดแบบ Vector
        orig_h, orig_w = orig_shape[:2]
        x_centers = filtered_output[:, 0]
        y_centers = filtered_output[:, 1]
        widths = filtered_output[:, 2]
        heights = filtered_output[:, 3]
        
        # คำนวณ x1, y1, x2, y2 ทีเดียวทั้งอาเรย์
        x1 = ((x_centers - widths / 2) * orig_w / self.input_size).astype(int)
        y1 = ((y_centers - heights / 2) * orig_h / self.input_size).astype(int)
        x2 = ((x_centers + widths / 2) * orig_w / self.input_size).astype(int)
        y2 = ((y_centers + heights / 2) * orig_h / self.input_size).astype(int)
        
        # จำกัดขอบเขตไม่ให้เกินขนาดภาพ
        x1 = np.clip(x1, 0, orig_w)
        y1 = np.clip(y1, 0, orig_h)
        x2 = np.clip(x2, 0, orig_w)
        y2 = np.clip(y2, 0, orig_h)
        
        # 5. เก็บผลลัพธ์
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


def detect_video(source, output_path=None, show_display=True, skip_frames=3,
                 low_res=False, cam_width=320, cam_height=240, input_size=INPUT_SIZE):
    """
    Detect helmets in video - High Performance Version
    
    Args:
        source: Video source (0 for webcam or video file path)
        output_path: Output video path (optional)
        show_display: Show real-time display
        skip_frames: Process every Nth frame (default: 3 for high FPS)
        low_res: Use lower camera resolution for even better performance
        cam_width: Camera capture width in pixels (default: 320)
        cam_height: Camera capture height in pixels (default: 240)
        input_size: Model input size (default: 320)
    """
    
    # ตรวจสอบว่าโมเดลมีอยู่ก่อนเริ่ม
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model file not found: {MODEL_PATH}")
        print(f"🔍 Please ensure the model file exists in the current directory")
        return
    
    print(f"🎯 Initializing helmet detection system...")
    print(f"📁 Model file: {MODEL_PATH}")
    
    # Initialize detector
    try:
        detector = HelmetDetector(MODEL_PATH, CONF_THRESHOLD, input_size)
        print(f"🚀 Detector initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize detector: {e}")
        return
    
    # Open video
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video source: {source}")
        return
    
    # ลดความละเอียดกล้องตั้งแต่ต้นเพื่อไม่ให้ Pi ต้องแบกภาพ 2K/FullHD
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print(f"📹 Camera resolution set to {cam_width}x{cam_height} for maximum performance")
    
    # Get video info (หลังจากเซ็ต resolution แล้ว)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    source_name = "Webcam" if source == 0 else source
    print(f"🎥 Video Source: {source_name}")
    print(f"📐 Resolution: {width}x{height}")
    print(f"🎬 FPS: {fps}")
    print(f"⚡ Frame Skip: {skip_frames} (processing every {skip_frames} frame(s))")
    if total_frames > 0:
        print(f"📊 Total Frames: {total_frames}\n")
    
    # Setup video writer
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"💾 Saving output to: {output_path}\n")
    
    # Statistics
    frame_count = 0
    processed_count = 0
    total_helmets = 0
    total_nohelmets = 0
    start_time = time.time()
    last_detections = []
    
    print("🚀 Starting HIGH PERFORMANCE detection... (Press 'q' to quit)\n")
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("\n✅ Video processing complete!")
                break
            
            frame_count += 1
            
            # Process only every Nth frame
            if frame_count % skip_frames == 0:
                processed_count += 1
                
                # Run detection
                detections = detector.detect(frame)
                last_detections = detections
                
                # Count detections
                helmets_in_frame = sum(1 for d in detections if d['class_name'] == 'helmet')
                nohelmets_in_frame = sum(1 for d in detections if d['class_name'] == 'nohelmet')
                
                total_helmets += helmets_in_frame
                total_nohelmets += nohelmets_in_frame
            
            # Draw detections (use last detections for skipped frames)
            for det in last_detections:
                x1, y1, x2, y2 = det['box']
                color = COLORS.get(det['class_name'], (255, 255, 255))
                
                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"{det['class_name']}: {det['confidence']:.2f}"
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
            
            helmets_now = sum(1 for d in last_detections if d['class_name'] == 'helmet')
            nohelmets_now = sum(1 for d in last_detections if d['class_name'] == 'nohelmet')
            cv2.putText(frame, f"H:{helmets_now} | NH:{nohelmets_now}",
                       (10, info_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Save frame
            if writer:
                writer.write(frame)
            
            # Display
            if show_display:
                cv2.imshow('Helmet Detection - Pi4 HIGH PERF', frame)
                
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
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        # Summary
        elapsed_time = time.time() - start_time
        print("\n" + "="*60)
        print("📊 Detection Summary")
        print("="*60)
        print(f"⏱️  Total Time: {elapsed_time:.2f} seconds")
        print(f"🎬 Total Frames: {frame_count}")
        print(f"⚡ Processed Frames: {processed_count}")
        print(f"📈 Processing FPS: {processed_count/elapsed_time:.2f}")
        print(f"✅ Total Helmets: {total_helmets}")
        print(f"❌ Total No Helmets: {total_nohelmets}")
        if output_path:
            print(f"💾 Output: {output_path}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Helmet Detection for Raspberry Pi 4 - High Performance')
    parser.add_argument('--source', type=str, default='0',
                       help='Video source: 0 for webcam or video file path')
    parser.add_argument('--output', type=str, default=None,
                       help='Output video path (optional)')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable display (for headless Pi)')
    parser.add_argument('--skip-frames', type=int, default=3,
                       help='Process every Nth frame (default: 3 for high performance)')
    parser.add_argument('--low-res', action='store_true',
                       help='Use low resolution camera mode (320x240) for maximum FPS')
    parser.add_argument('--cam-width', type=int, default=320,
                       help='Camera capture width in pixels (default: 320)')
    parser.add_argument('--cam-height', type=int, default=240,
                       help='Camera capture height in pixels (default: 240)')
    parser.add_argument('--input-size', type=int, default=INPUT_SIZE,
                       choices=[192, 256, 320, 416, 640],
                       help='Model input size: 192=fastest, 256=ultra, 320=default, 640=accurate')
    
    args = parser.parse_args()
    
    # Convert source to int if digit
    source = args.source
    if source.isdigit():
        source = int(source)
    
    # Run detection
    detect_video(
        source=source,
        output_path=args.output,
        show_display=not args.no_display,
        skip_frames=args.skip_frames,
        low_res=args.low_res,
        cam_width=args.cam_width,
        cam_height=args.cam_height,
        input_size=args.input_size
    )


if __name__ == "__main__":
    main()
