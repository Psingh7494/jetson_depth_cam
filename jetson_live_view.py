#!/usr/bin/env python3
"""
Intel RealSense D435 Live View Script - Optimized for NVIDIA Jetson AGX Xavier
Displays live color and depth streams from the RealSense camera with GPU acceleration
Press 'q' to quit the application

Features:
- GPU-accelerated image processing using OpenCV CUDA
- Optimized memory management for Jetson
- Higher resolution support (up to 1280x720)
- Real-time FPS monitoring
- Hardware-specific optimizations for Jetson AGX Xavier
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import sys
import time
import gc
import argparse

class JetsonLiveView:
    def __init__(self, width=640, height=480, fps=30, enable_gpu=True):
        self.width = width
        self.height = height
        self.fps = fps
        self.enable_gpu = enable_gpu
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.align = None
        
        # Performance monitoring
        self.frame_count = 0
        self.start_time = time.time()
        self.last_fps_time = time.time()
        self.last_fps_count = 0
        
        # GPU availability check
        self.gpu_available = False
        if self.enable_gpu:
            try:
                # Check if OpenCV was compiled with CUDA support
                if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    self.gpu_available = True
                    print(f"GPU acceleration enabled - {cv2.cuda.getCudaEnabledDeviceCount()} CUDA device(s) detected")
                else:
                    print("GPU acceleration requested but no CUDA devices available")
            except:
                print("GPU acceleration requested but OpenCV CUDA support not available")
        
        self.setup_camera()
        
    def setup_camera(self):
        """Configure RealSense camera with Jetson-optimized settings"""
        # Get device product line for setting a supporting resolution
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
            sys.exit(1)
        
        # Configure streams with Jetson-optimized resolutions
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        
        print(f"Camera configured: {self.width}x{self.height} @ {self.fps}fps")
        print(f"Device: {device_product_line}")
        
    def process_frame_gpu(self, color_image, depth_image):
        """GPU-accelerated frame processing"""
        try:
            # Upload to GPU
            gpu_color = cv2.cuda_GpuMat()
            gpu_depth = cv2.cuda_GpuMat()
            gpu_color.upload(color_image)
            gpu_depth.upload(depth_image)
            
            # Convert depth to 8-bit on GPU
            gpu_depth_8bit = cv2.cuda_GpuMat()
            cv2.cuda.convertScaleAbs(gpu_depth, gpu_depth_8bit, alpha=0.03)
            
            # Apply colormap on GPU
            gpu_depth_colormap = cv2.cuda_GpuMat()
            cv2.cuda.applyColorMap(gpu_depth_8bit, gpu_depth_colormap, cv2.COLORMAP_JET)
            
            # Download from GPU
            depth_colormap = gpu_depth_colormap.download()
            
            return color_image, depth_colormap
            
        except Exception as e:
            print(f"GPU processing failed, falling back to CPU: {e}")
            return self.process_frame_cpu(color_image, depth_image)
    
    def process_frame_cpu(self, color_image, depth_image):
        """CPU-based frame processing fallback"""
        # Apply colormap on depth image
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), 
            cv2.COLORMAP_JET
        )
        return color_image, depth_colormap
    
    def create_display_image(self, color_image, depth_colormap):
        """Create side-by-side display image with performance optimizations"""
        # Ensure both images have the same height
        if color_image.shape[:2] != depth_colormap.shape[:2]:
            depth_colormap = cv2.resize(
                depth_colormap, 
                (color_image.shape[1], color_image.shape[0]), 
                interpolation=cv2.INTER_LINEAR
            )
        
        # Create horizontal stack
        display_image = np.hstack((color_image, depth_colormap))
        
        # Add text overlays
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        color = (255, 255, 255)
        
        # Labels
        cv2.putText(display_image, 'Color', (10, 30), font, font_scale, color, thickness)
        cv2.putText(display_image, 'Depth', (display_image.shape[1]//2 + 10, 30), 
                   font, font_scale, color, thickness)
        
        # Performance info
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:  # Update FPS every second
            frames_in_second = self.frame_count - self.last_fps_count
            fps = frames_in_second / (current_time - self.last_fps_time)
            self.current_fps = fps
            self.last_fps_time = current_time
            self.last_fps_count = self.frame_count
        
        if hasattr(self, 'current_fps'):
            fps_text = f"FPS: {self.current_fps:.1f}"
            cv2.putText(display_image, fps_text, (10, display_image.shape[0] - 20), 
                       font, 0.6, (0, 255, 0), 2)
        
        # GPU status
        gpu_text = "GPU: ON" if self.gpu_available else "GPU: OFF"
        gpu_color = (0, 255, 0) if self.gpu_available else (0, 0, 255)
        cv2.putText(display_image, gpu_text, (10, display_image.shape[0] - 45), 
                   font, 0.6, gpu_color, 2)
        
        return display_image
    
    def run(self):
        """Main live view loop with Jetson optimizations"""
        try:
            # Start streaming
            profile = self.pipeline.start(self.config)
            
            # Get depth sensor and set some optimizations
            depth_sensor = profile.get_device().first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()
            
            # Create alignment object to align depth frames to color frames
            self.align = rs.align(rs.stream.color)
            
            print("RealSense D435 Live View Started (Jetson Optimized)")
            print("Controls:")
            print("  'q' - Quit")
            print("  'g' - Toggle GPU acceleration (if available)")
            print("  'f' - Show detailed frame info")
            print("  'r' - Reset FPS counter")
            
            # Pre-allocate window
            cv2.namedWindow('RealSense Jetson Live View', cv2.WINDOW_AUTOSIZE)
            
            show_frame_info = False
            
            while True:
                # Wait for coherent pair of frames
                frames = self.pipeline.wait_for_frames()
                
                # Align depth frame to color frame
                aligned_frames = self.align.process(frames)
                
                # Get aligned frames
                aligned_depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                
                if not aligned_depth_frame or not color_frame:
                    continue
                
                # Convert images to numpy arrays
                depth_image = np.asanyarray(aligned_depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                
                # Process frames (GPU or CPU)
                if self.gpu_available and self.enable_gpu:
                    color_processed, depth_colormap = self.process_frame_gpu(color_image, depth_image)
                else:
                    color_processed, depth_colormap = self.process_frame_cpu(color_image, depth_image)
                
                # Create display image
                display_image = self.create_display_image(color_processed, depth_colormap)
                
                # Add frame info if requested
                if show_frame_info:
                    info_y = 70
                    cv2.putText(display_image, f"Frame: {self.frame_count}", 
                               (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(display_image, f"Depth Scale: {depth_scale:.4f}", 
                               (10, info_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(display_image, f"Resolution: {self.width}x{self.height}", 
                               (10, info_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Display the image
                cv2.imshow('RealSense Jetson Live View', display_image)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('g') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    self.enable_gpu = not self.enable_gpu
                    print(f"GPU acceleration: {'ON' if self.enable_gpu else 'OFF'}")
                elif key == ord('f'):
                    show_frame_info = not show_frame_info
                    print(f"Frame info display: {'ON' if show_frame_info else 'OFF'}")
                elif key == ord('r'):
                    self.frame_count = 0
                    self.start_time = time.time()
                    self.last_fps_time = time.time()
                    self.last_fps_count = 0
                    print("FPS counter reset")
                
                self.frame_count += 1
                
                # Periodic garbage collection to manage memory on Jetson
                if self.frame_count % 300 == 0:  # Every ~10 seconds at 30fps
                    gc.collect()
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            # Cleanup
            self.pipeline.stop()
            cv2.destroyAllWindows()
            
            # Final statistics
            total_time = time.time() - self.start_time
            avg_fps = self.frame_count / total_time if total_time > 0 else 0
            print(f"\nSession Statistics:")
            print(f"Total frames: {self.frame_count}")
            print(f"Total time: {total_time:.2f} seconds")
            print(f"Average FPS: {avg_fps:.2f}")
            print("Live view stopped")

def main():
    parser = argparse.ArgumentParser(description='RealSense Live View - Jetson AGX Xavier Optimized')
    parser.add_argument('-w', '--width', type=int, default=640, 
                       choices=[640, 848, 1280], 
                       help='Frame width (default: 640)')
    parser.add_argument('-h', '--height', type=int, default=480, 
                       choices=[480, 480, 720], 
                       help='Frame height (default: 480)')
    parser.add_argument('-f', '--fps', type=int, default=30, 
                       choices=[15, 30, 60], 
                       help='Frames per second (default: 30)')
    parser.add_argument('--no-gpu', action='store_true', 
                       help='Disable GPU acceleration')
    parser.add_argument('--preset', type=str, 
                       choices=['low', 'medium', 'high', 'ultra'],
                       help='Quality presets: low=640x480@30, medium=848x480@30, high=1280x720@30, ultra=1280x720@60')
    
    args = parser.parse_args()
    
    # Handle presets
    if args.preset:
        presets = {
            'low': (640, 480, 30),
            'medium': (848, 480, 30), 
            'high': (1280, 720, 30),
            'ultra': (1280, 720, 60)
        }
        args.width, args.height, args.fps = presets[args.preset]
    
    # Validate resolution combinations
    valid_resolutions = [(640, 480), (848, 480), (1280, 720)]
    if (args.width, args.height) not in valid_resolutions:
        print(f"Error: Invalid resolution {args.width}x{args.height}")
        print(f"Valid resolutions: {valid_resolutions}")
        return
    
    print(f"Starting Jetson Live View...")
    print(f"Resolution: {args.width}x{args.height} @ {args.fps}fps")
    print(f"GPU Acceleration: {'Disabled' if args.no_gpu else 'Enabled (if available)'}")
    
    live_view = JetsonLiveView(
        width=args.width, 
        height=args.height, 
        fps=args.fps, 
        enable_gpu=not args.no_gpu
    )
    live_view.run()

if __name__ == "__main__":
    main()
