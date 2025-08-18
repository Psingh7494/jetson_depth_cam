#!/usr/bin/env python3
"""
Intel RealSense D435 Timed Recording Script - Optimized for NVIDIA Jetson AGX Xavier
Records color and depth streams for a specified duration with GPU acceleration
Saves both .bag file and individual frame data to the 'data' folder

Features:
- GPU-accelerated image processing using OpenCV CUDA
- Optimized memory management for Jetson
- High-resolution recording support (up to 1280x720)
- Real-time compression options
- Hardware-specific optimizations for Jetson AGX Xavier
- Intelligent storage management
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import os
import json
import time
import argparse
import threading
import queue
import gc
from datetime import datetime, timedelta
from pathlib import Path

class JetsonTimedRecorder:
    def __init__(self, data_folder="data", duration_seconds=30, width=640, height=480, fps=30, 
                 enable_gpu=True, compress_images=False, save_raw_depth=True):
        self.data_folder = data_folder
        self.duration_seconds = duration_seconds
        self.width = width
        self.height = height
        self.fps = fps
        self.enable_gpu = enable_gpu
        self.compress_images = compress_images
        self.save_raw_depth = save_raw_depth
        
        # Pipeline setup
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.align = None
        
        # Recording state
        self.recording = False
        self.session_folder = None
        self.frame_count = 0
        self.start_time = None
        self.end_time = None
        
        # Performance monitoring
        self.frames_processed = 0
        self.frames_saved = 0
        self.save_queue = queue.Queue(maxsize=60)  # Buffer for 2 seconds at 30fps
        self.save_thread = None
        self.save_thread_running = False
        
        # GPU setup
        self.gpu_available = False
        if self.enable_gpu:
            try:
                if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    self.gpu_available = True
                    print(f"GPU acceleration enabled - {cv2.cuda.getCudaEnabledDeviceCount()} CUDA device(s) detected")
                else:
                    print("GPU acceleration requested but no CUDA devices available")
            except:
                print("GPU acceleration requested but OpenCV CUDA support not available")
        
        # Compression settings
        if self.compress_images:
            self.png_compression = [cv2.IMWRITE_PNG_COMPRESSION, 6]  # Medium compression
            self.jpg_quality = [cv2.IMWRITE_JPEG_QUALITY, 85]  # High quality JPEG
        else:
            self.png_compression = [cv2.IMWRITE_PNG_COMPRESSION, 0]  # No compression
            self.jpg_quality = [cv2.IMWRITE_JPEG_QUALITY, 100]  # Maximum quality
        
        # Ensure data folder exists
        Path(self.data_folder).mkdir(parents=True, exist_ok=True)
        
        self.setup_camera()
        
    def setup_camera(self):
        """Configure RealSense camera with Jetson-optimized settings"""
        # Get device info
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))
        
        # Check for RGB camera
        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("Error: RealSense camera with RGB sensor required")
            exit(1)
        
        # Configure streams
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        
        print(f"Camera configured: {self.width}x{self.height} @ {self.fps}fps")
        print(f"Device: {device_product_line}")
        
    def start_recording_session(self):
        """Start a new timed recording session"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_folder = Path(self.data_folder) / f"jetson_recording_{timestamp}"
        self.session_folder.mkdir(parents=True, exist_ok=True)
        
        # Create subfolders
        (self.session_folder / "color").mkdir(exist_ok=True)
        (self.session_folder / "depth").mkdir(exist_ok=True)
        if self.save_raw_depth:
            (self.session_folder / "depth_raw").mkdir(exist_ok=True)
        
        # Configure bag recording
        bag_filename = self.session_folder / f"jetson_recording_{timestamp}.bag"
        self.config.enable_record_to_file(str(bag_filename))
        
        # Reset counters
        self.recording = True
        self.frame_count = 0
        self.frames_processed = 0
        self.frames_saved = 0
        self.start_time = time.time()
        self.end_time = self.start_time + self.duration_seconds
        
        # Start save thread
        self.save_thread_running = True
        self.save_thread = threading.Thread(target=self._save_worker, daemon=True)
        self.save_thread.start()
        
        # Create metadata
        metadata = {
            "session_start": timestamp,
            "duration_seconds": self.duration_seconds,
            "estimated_end_time": (datetime.now() + timedelta(seconds=self.duration_seconds)).strftime("%Y%m%d_%H%M%S"),
            "bag_file": f"jetson_recording_{timestamp}.bag",
            "format": {
                "color": f"BGR8 {self.width}x{self.height} @{self.fps}fps",
                "depth": f"Z16 {self.width}x{self.height} @{self.fps}fps"
            },
            "jetson_optimizations": {
                "gpu_acceleration": self.gpu_available,
                "image_compression": self.compress_images,
                "save_raw_depth": self.save_raw_depth,
                "async_saving": True
            }
        }
        
        with open(self.session_folder / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=4)
            
        print(f"Jetson recording started: {self.session_folder}")
        print(f"Recording duration: {self.duration_seconds} seconds")
        print(f"GPU acceleration: {'ON' if self.gpu_available else 'OFF'}")
        print(f"Image compression: {'ON' if self.compress_images else 'OFF'}")
        
    def stop_recording_session(self):
        """Stop the current recording session"""
        if not self.recording:
            return
            
        self.recording = False
        actual_duration = time.time() - self.start_time
        
        # Stop save thread
        self.save_thread_running = False
        if self.save_thread and self.save_thread.is_alive():
            self.save_thread.join(timeout=10)
        
        # Process remaining items in queue
        while not self.save_queue.empty():
            try:
                save_data = self.save_queue.get_nowait()
                self._save_frame_data(save_data)
                self.frames_saved += 1
            except queue.Empty:
                break
        
        # Update metadata
        metadata_path = self.session_folder / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            metadata.update({
                "session_end": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "actual_duration_seconds": round(actual_duration, 2),
                "total_frames_processed": self.frames_processed,
                "total_frames_saved": self.frames_saved,
                "average_fps": round(self.frames_processed / actual_duration, 2) if actual_duration > 0 else 0,
                "save_efficiency": round((self.frames_saved / self.frames_processed) * 100, 1) if self.frames_processed > 0 else 0
            })
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
        
        print(f"\nRecording completed!")
        print(f"Duration: {actual_duration:.2f} seconds")
        print(f"Frames processed: {self.frames_processed}")
        print(f"Frames saved: {self.frames_saved}")
        print(f"Average FPS: {self.frames_processed / actual_duration:.2f}" if actual_duration > 0 else "N/A")
        print(f"Save efficiency: {(self.frames_saved / self.frames_processed) * 100:.1f}%" if self.frames_processed > 0 else "N/A")
        print(f"Data saved to: {self.session_folder}")
        
    def process_frame_gpu(self, color_image, depth_image):
        """GPU-accelerated frame processing"""
        try:
            # Upload to GPU
            gpu_depth = cv2.cuda_GpuMat()
            gpu_depth.upload(depth_image)
            
            # Convert depth to 8-bit on GPU
            gpu_depth_8bit = cv2.cuda_GpuMat()
            cv2.cuda.convertScaleAbs(gpu_depth, gpu_depth_8bit, alpha=0.03)
            
            # Apply colormap on GPU
            gpu_depth_colormap = cv2.cuda_GpuMat()
            cv2.cuda.applyColorMap(gpu_depth_8bit, gpu_depth_colormap, cv2.COLORMAP_JET)
            
            # Download from GPU
            depth_colormap = gpu_depth_colormap.download()
            
            return depth_colormap
            
        except Exception as e:
            print(f"GPU processing failed, falling back to CPU: {e}")
            return self.process_frame_cpu(depth_image)
    
    def process_frame_cpu(self, depth_image):
        """CPU-based frame processing fallback"""
        return cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    
    def _save_worker(self):
        """Background thread for saving frames to disk"""
        while self.save_thread_running or not self.save_queue.empty():
            try:
                save_data = self.save_queue.get(timeout=1.0)
                self._save_frame_data(save_data)
                self.frames_saved += 1
                self.save_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in save worker: {e}")
                
    def _save_frame_data(self, save_data):
        """Save individual frame data to disk"""
        frame_name, color_image, depth_colormap, depth_image, frame_metadata = save_data
        
        try:
            # Save color image
            color_path = self.session_folder / "color" / f"{frame_name}.png"
            cv2.imwrite(str(color_path), color_image, self.png_compression)
            
            # Save depth colormap
            depth_path = self.session_folder / "depth" / f"{frame_name}.png"
            cv2.imwrite(str(depth_path), depth_colormap, self.png_compression)
            
            # Save raw depth data if enabled
            if self.save_raw_depth:
                depth_raw_path = self.session_folder / "depth_raw" / f"{frame_name}.npy"
                np.save(str(depth_raw_path), depth_image)
            
            # Save frame metadata
            metadata_path = self.session_folder / f"{frame_name}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(frame_metadata, f, indent=2)
                
        except Exception as e:
            print(f"Error saving frame {frame_name}: {e}")
    
    def queue_frame_for_saving(self, color_image, depth_image, depth_colormap, aligned_depth_frame):
        """Queue frame data for background saving"""
        if not self.recording:
            return False
            
        frame_name = f"frame_{self.frame_count:06d}"
        
        # Create frame metadata
        frame_metadata = {
            "frame_number": self.frame_count,
            "timestamp": datetime.now().isoformat(),
            "time_since_start": time.time() - self.start_time,
            "depth_scale": aligned_depth_frame.get_units(),
            "resolution": f"{self.width}x{self.height}",
            "fps": self.fps
        }
        
        # Prepare save data
        save_data = (frame_name, color_image.copy(), depth_colormap.copy(), 
                    depth_image.copy() if self.save_raw_depth else None, frame_metadata)
        
        # Try to add to queue (non-blocking)
        try:
            self.save_queue.put_nowait(save_data)
            return True
        except queue.Full:
            print(f"Warning: Save queue full, dropping frame {self.frame_count}")
            return False
    
    def get_time_remaining(self):
        """Get remaining recording time in seconds"""
        if not self.recording or not self.start_time:
            return 0
        return max(0, self.end_time - time.time())
        
    def is_recording_complete(self):
        """Check if recording duration has been reached"""
        if not self.recording:
            return False
        return time.time() >= self.end_time
    
    def run_with_preview(self):
        """Run recording with live preview"""
        try:
            # Start streaming
            profile = self.pipeline.start(self.config)
            
            # Create alignment object
            self.align = rs.align(rs.stream.color)
            
            print(f"RealSense D435 Jetson Timed Recording Mode")
            print(f"Recording Duration: {self.duration_seconds} seconds")
            print(f"Resolution: {self.width}x{self.height} @ {self.fps}fps")
            print("Controls:")
            print("  'r' - Start recording")
            print("  'q' - Quit application")
            print("  SPACE - Stop recording early")
            print("  'g' - Toggle GPU acceleration (if available)")
            print("  's' - Show statistics")
            print("\nPress 'r' to start recording or 'q' to quit...")
            
            # Performance tracking
            last_fps_time = time.time()
            last_fps_count = 0
            current_fps = 0.0
            
            while True:
                # Wait for frames
                frames = self.pipeline.wait_for_frames()
                
                # Align frames
                aligned_frames = self.align.process(frames)
                aligned_depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                
                if not aligned_depth_frame or not color_frame:
                    continue
                
                # Convert to numpy arrays
                depth_image = np.asanyarray(aligned_depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                
                # Check if recording should stop
                if self.recording and self.is_recording_complete():
                    self.stop_recording_session()
                    break
                
                # Process depth image
                if self.gpu_available and self.enable_gpu:
                    depth_colormap = self.process_frame_gpu(color_image, depth_image)
                else:
                    depth_colormap = self.process_frame_cpu(depth_image)
                
                # Queue frame for saving if recording
                if self.recording:
                    self.queue_frame_for_saving(color_image, depth_image, depth_colormap, aligned_depth_frame)
                    self.frame_count += 1
                
                self.frames_processed += 1
                
                # Create display image
                if depth_colormap.shape[:2] != color_image.shape[:2]:
                    depth_colormap = cv2.resize(depth_colormap, (color_image.shape[1], color_image.shape[0]))
                
                display_image = np.hstack((color_image, depth_colormap))
                
                # Calculate FPS
                current_time = time.time()
                if current_time - last_fps_time >= 1.0:
                    frames_in_second = self.frames_processed - last_fps_count
                    current_fps = frames_in_second / (current_time - last_fps_time)
                    last_fps_time = current_time
                    last_fps_count = self.frames_processed
                
                # Add status overlay
                font = cv2.FONT_HERSHEY_SIMPLEX
                if self.recording:
                    time_remaining = self.get_time_remaining()
                    status_text = f"RECORDING - {time_remaining:.1f}s left"
                    status_color = (0, 0, 255)
                    cv2.putText(display_image, f"Frame: {self.frame_count}", 
                               (10, display_image.shape[0] - 60), font, 0.6, (255, 255, 255), 2)
                    cv2.putText(display_image, f"Queue: {self.save_queue.qsize()}", 
                               (10, display_image.shape[0] - 40), font, 0.6, (255, 255, 255), 2)
                else:
                    status_text = "READY TO RECORD"
                    status_color = (0, 255, 0)
                
                cv2.putText(display_image, status_text, (10, 30), font, 1, status_color, 2)
                cv2.putText(display_image, 'Color', (10, 60), font, 0.7, (255, 255, 255), 2)
                cv2.putText(display_image, 'Depth', (display_image.shape[1]//2 + 10, 60), font, 0.7, (255, 255, 255), 2)
                cv2.putText(display_image, f"FPS: {current_fps:.1f}", 
                           (10, display_image.shape[0] - 20), font, 0.6, (0, 255, 0), 2)
                
                # GPU status
                gpu_text = "GPU: ON" if (self.gpu_available and self.enable_gpu) else "GPU: OFF"
                gpu_color = (0, 255, 0) if (self.gpu_available and self.enable_gpu) else (0, 0, 255)
                cv2.putText(display_image, gpu_text, (display_image.shape[1] - 120, display_image.shape[0] - 20), 
                           font, 0.6, gpu_color, 2)
                
                # Display
                cv2.namedWindow('RealSense Jetson Timed Recording', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('RealSense Jetson Timed Recording', display_image)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r') and not self.recording:
                    # Start recording
                    self.pipeline.stop()
                    self.config = rs.config()
                    self.setup_camera()
                    self.start_recording_session()
                    self.pipeline.start(self.config)
                elif key == ord(' ') and self.recording:  # Spacebar to stop early
                    print("Recording stopped early by user")
                    self.stop_recording_session()
                    break
                elif key == ord('g') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    self.enable_gpu = not self.enable_gpu
                    print(f"GPU acceleration: {'ON' if self.enable_gpu else 'OFF'}")
                elif key == ord('s'):
                    print(f"\n=== Current Statistics ===")
                    print(f"Frames processed: {self.frames_processed}")
                    if self.recording:
                        print(f"Frames recorded: {self.frame_count}")
                        print(f"Frames saved: {self.frames_saved}")
                        print(f"Save queue size: {self.save_queue.qsize()}")
                    print(f"Current FPS: {current_fps:.2f}")
                    print(f"GPU acceleration: {'ON' if (self.gpu_available and self.enable_gpu) else 'OFF'}")
                    print("========================\n")
                
                # Periodic garbage collection
                if self.frames_processed % 300 == 0:
                    gc.collect()
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            if self.recording:
                self.stop_recording_session()
            self.pipeline.stop()
            cv2.destroyAllWindows()
            print("Jetson timed recording session ended")
    
    def run_headless(self):
        """Run recording without preview (headless mode)"""
        try:
            self.start_recording_session()
            profile = self.pipeline.start(self.config)
            
            # Create alignment object
            self.align = rs.align(rs.stream.color)
            
            print(f"Starting headless Jetson recording for {self.duration_seconds} seconds...")
            
            last_progress_time = time.time()
            
            while self.recording and not self.is_recording_complete():
                # Wait for frames
                frames = self.pipeline.wait_for_frames()
                
                # Align frames
                aligned_frames = self.align.process(frames)
                aligned_depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                
                if not aligned_depth_frame or not color_frame:
                    continue
                
                # Convert to numpy arrays
                depth_image = np.asanyarray(aligned_depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                
                # Process depth image
                if self.gpu_available and self.enable_gpu:
                    depth_colormap = self.process_frame_gpu(color_image, depth_image)
                else:
                    depth_colormap = self.process_frame_cpu(depth_image)
                
                # Queue frame for saving
                self.queue_frame_for_saving(color_image, depth_image, depth_colormap, aligned_depth_frame)
                self.frame_count += 1
                self.frames_processed += 1
                
                # Print progress every 5 seconds
                current_time = time.time()
                if current_time - last_progress_time >= 5.0:
                    time_remaining = self.get_time_remaining()
                    queue_size = self.save_queue.qsize()
                    print(f"Recording... {time_remaining:.1f}s remaining, {self.frame_count} frames captured, queue: {queue_size}")
                    last_progress_time = current_time
                
                # Periodic garbage collection
                if self.frames_processed % 150 == 0:
                    gc.collect()
            
            self.stop_recording_session()
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            if self.recording:
                self.stop_recording_session()
            self.pipeline.stop()
            print("Headless Jetson recording completed")

def main():
    parser = argparse.ArgumentParser(description='Intel RealSense D435 Jetson Timed Recording')
    parser.add_argument('-d', '--duration', type=int, default=30, 
                       help='Recording duration in seconds (default: 30)')
    parser.add_argument('-o', '--output', type=str, default='data',
                       help='Output folder for recordings (default: data)')
    parser.add_argument('-w', '--width', type=int, default=640, 
                       choices=[640, 848, 1280], 
                       help='Frame width (default: 640)')
    parser.add_argument('-h', '--height', type=int, default=480, 
                       choices=[480, 480, 720], 
                       help='Frame height (default: 480)')
    parser.add_argument('-f', '--fps', type=int, default=30, 
                       choices=[15, 30, 60], 
                       help='Frames per second (default: 30)')
    parser.add_argument('--headless', action='store_true',
                       help='Run without GUI preview (headless mode)')
    parser.add_argument('--no-gpu', action='store_true', 
                       help='Disable GPU acceleration')
    parser.add_argument('--compress', action='store_true',
                       help='Enable image compression to save storage space')
    parser.add_argument('--no-raw-depth', action='store_true',
                       help='Skip saving raw depth data (.npy files)')
    parser.add_argument('--preset', type=str, 
                       choices=['quick', 'short', 'medium', 'long', 'hd', 'ultra'],
                       help='Presets: quick=10s@640x480, short=30s@640x480, medium=120s@848x480, long=300s@848x480, hd=120s@1280x720, ultra=300s@1280x720@60fps')
    
    args = parser.parse_args()
    
    # Handle presets
    if args.preset:
        presets = {
            'quick': (10, 640, 480, 30),
            'short': (30, 640, 480, 30),
            'medium': (120, 848, 480, 30), 
            'long': (300, 848, 480, 30),
            'hd': (120, 1280, 720, 30),
            'ultra': (300, 1280, 720, 60)
        }
        args.duration, args.width, args.height, args.fps = presets[args.preset]
    
    # Validate parameters
    if args.duration <= 0:
        print("Error: Duration must be greater than 0 seconds")
        return
    
    if args.duration > 3600:  # 1 hour limit
        print("Warning: Recording duration is longer than 1 hour")
        confirm = input("Continue? (y/n): ")
        if confirm.lower() != 'y':
            return
    
    # Validate resolution combinations
    valid_resolutions = [(640, 480), (848, 480), (1280, 720)]
    if (args.width, args.height) not in valid_resolutions:
        print(f"Error: Invalid resolution {args.width}x{args.height}")
        print(f"Valid resolutions: {valid_resolutions}")
        return
    
    print(f"Initializing Jetson timed recorder...")
    print(f"Duration: {args.duration} seconds ({args.duration/60:.1f} minutes)")
    print(f"Resolution: {args.width}x{args.height} @ {args.fps}fps")
    print(f"Output folder: {args.output}")
    print(f"Mode: {'Headless' if args.headless else 'With preview'}")
    print(f"GPU acceleration: {'Disabled' if args.no_gpu else 'Enabled (if available)'}")
    print(f"Image compression: {'Enabled' if args.compress else 'Disabled'}")
    print(f"Raw depth saving: {'Disabled' if args.no_raw_depth else 'Enabled'}")
    
    recorder = JetsonTimedRecorder(
        data_folder=args.output, 
        duration_seconds=args.duration,
        width=args.width,
        height=args.height,
        fps=args.fps,
        enable_gpu=not args.no_gpu,
        compress_images=args.compress,
        save_raw_depth=not args.no_raw_depth
    )
    
    if args.headless:
        recorder.run_headless()
    else:
        recorder.run_with_preview()

if __name__ == "__main__":
    main()
