#!/usr/bin/env python3
"""
Intel RealSense D435 Timed Recording Script - Jetson AGX Xavier (Python 3.7)
Simple timed recording optimized for Jetson AGX Xavier
Records color and depth streams for a specified duration
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import os
import json
import time
import argparse
import gc
from datetime import datetime, timedelta

class JetsonSimpleRecorder:
    def __init__(self, data_folder="data", duration_seconds=30):
        self.data_folder = data_folder
        self.duration_seconds = duration_seconds
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.recording = False
        self.session_folder = None
        self.frame_count = 0
        self.start_time = None
        self.end_time = None
        
        # Ensure data folder exists
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)
        
        # Configure streams
        self.setup_streams()
        
    def setup_streams(self):
        """Configure depth and color streams for Jetson"""
        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("The demo requires Depth camera with Color sensor")
            exit(0)

        # Configure streams - optimized for Jetson AGX Xavier
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        if device_product_line == 'L500':
            self.config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
        else:
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            
    def start_recording_session(self):
        """Start a new timed recording session"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_folder = os.path.join(self.data_folder, f"jetson_recording_{timestamp}")
        os.makedirs(self.session_folder, exist_ok=True)
        
        # Create subfolders for different data types
        os.makedirs(os.path.join(self.session_folder, "color"), exist_ok=True)
        os.makedirs(os.path.join(self.session_folder, "depth"), exist_ok=True)
        os.makedirs(os.path.join(self.session_folder, "depth_raw"), exist_ok=True)
        
        # Configure bag recording
        bag_filename = os.path.join(self.session_folder, f"jetson_recording_{timestamp}.bag")
        self.config.enable_record_to_file(bag_filename)
        
        self.recording = True
        self.frame_count = 0
        self.start_time = time.time()
        self.end_time = self.start_time + self.duration_seconds
        
        # Create metadata file
        metadata = {
            "session_start": timestamp,
            "duration_seconds": self.duration_seconds,
            "estimated_end_time": (datetime.now() + timedelta(seconds=self.duration_seconds)).strftime("%Y%m%d_%H%M%S"),
            "bag_file": f"jetson_recording_{timestamp}.bag",
            "platform": "Jetson AGX Xavier",
            "format": {
                "color": "BGR8 640x480 @30fps",
                "depth": "Z16 640x480 @30fps"
            }
        }
        
        with open(os.path.join(self.session_folder, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=4)
            
        print(f"Jetson recording started: {self.session_folder}")
        print(f"Recording duration: {self.duration_seconds} seconds")
        
    def stop_recording_session(self):
        """Stop the current recording session"""
        if self.recording:
            self.recording = False
            actual_duration = time.time() - self.start_time
            
            # Update metadata with session end info
            metadata_path = os.path.join(self.session_folder, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                metadata["session_end"] = datetime.now().strftime("%Y%m%d_%H%M%S")
                metadata["actual_duration_seconds"] = round(actual_duration, 2)
                metadata["total_frames"] = self.frame_count
                metadata["average_fps"] = round(self.frame_count / actual_duration, 2) if actual_duration > 0 else 0
                
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=4)
            
            print(f"Recording completed!")
            print(f"Duration: {actual_duration:.2f} seconds")
            print(f"Total frames: {self.frame_count}")
            print(f"Average FPS: {self.frame_count / actual_duration:.2f}" if actual_duration > 0 else "N/A")
            print(f"Data saved to: {self.session_folder}")
            
    def save_frame_data(self, color_image, depth_image, aligned_depth_frame):
        """Save individual frame data"""
        if not self.recording:
            return
            
        frame_name = f"frame_{self.frame_count:06d}"
        
        # Save color image
        color_path = os.path.join(self.session_folder, "color", f"{frame_name}.png")
        cv2.imwrite(color_path, color_image)
        
        # Save depth colormap
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        depth_path = os.path.join(self.session_folder, "depth", f"{frame_name}.png")
        cv2.imwrite(depth_path, depth_colormap)
        
        # Save raw depth data as numpy array
        depth_raw_path = os.path.join(self.session_folder, "depth_raw", f"{frame_name}.npy")
        np.save(depth_raw_path, depth_image)
        
        # Save frame metadata
        frame_metadata = {
            "frame_number": self.frame_count,
            "timestamp": datetime.now().isoformat(),
            "time_since_start": time.time() - self.start_time,
            "depth_scale": aligned_depth_frame.get_units()
        }
        
        frame_metadata_path = os.path.join(self.session_folder, f"{frame_name}_metadata.json")
        with open(frame_metadata_path, 'w') as f:
            json.dump(frame_metadata, f, indent=4)
            
        self.frame_count += 1
        
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
            align_to = rs.stream.color
            align = rs.align(align_to)
            
            print(f"RealSense D435 Jetson Timed Recording")
            print(f"Recording Duration: {self.duration_seconds} seconds")
            print("Controls:")
            print("  'r' - Start recording")
            print("  'q' - Quit application")
            print("  SPACE - Stop recording early")
            print("\nPress 'r' to start recording or 'q' to quit...")
            
            while True:
                # Wait for frames
                frames = self.pipeline.wait_for_frames()
                
                # Align frames
                aligned_frames = align.process(frames)
                aligned_depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                
                if not aligned_depth_frame or not color_frame:
                    continue
                
                # Convert to numpy arrays
                depth_image = np.asanyarray(aligned_depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                
                # Check if recording should automatically stop
                if self.recording and self.is_recording_complete():
                    self.stop_recording_session()
                    break
                
                # Save frame data if recording
                if self.recording:
                    self.save_frame_data(color_image, depth_image, aligned_depth_frame)
                
                # Create display image
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                
                depth_colormap_dim = depth_colormap.shape
                color_colormap_dim = color_image.shape
                
                if depth_colormap_dim != color_colormap_dim:
                    resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
                    images = np.hstack((resized_color_image, depth_colormap))
                else:
                    images = np.hstack((color_image, depth_colormap))
                
                # Add status overlay
                if self.recording:
                    time_remaining = self.get_time_remaining()
                    status_text = f"RECORDING - {time_remaining:.1f}s left"
                    status_color = (0, 0, 255)
                    cv2.putText(images, f"Frame: {self.frame_count}", (10, images.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                else:
                    status_text = "READY TO RECORD"
                    status_color = (0, 255, 0)
                
                cv2.putText(images, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
                cv2.putText(images, 'Color', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(images, 'Depth', (images.shape[1]//2 + 10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Display
                cv2.namedWindow('RealSense Jetson Timed Recording', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('RealSense Jetson Timed Recording', images)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r') and not self.recording:
                    # Start recording
                    self.pipeline.stop()
                    self.config = rs.config()
                    self.setup_streams()
                    self.start_recording_session()
                    self.pipeline.start(self.config)
                elif key == ord(' ') and self.recording:  # Spacebar to stop early
                    print("Recording stopped early by user")
                    self.stop_recording_session()
                    break
                
                # Periodic memory cleanup for Jetson
                if self.frame_count % 300 == 0:
                    gc.collect()
                    
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
            # Start streaming
            self.start_recording_session()
            profile = self.pipeline.start(self.config)
            
            # Create alignment object
            align_to = rs.stream.color
            align = rs.align(align_to)
            
            print(f"Starting headless Jetson recording for {self.duration_seconds} seconds...")
            
            while self.recording and not self.is_recording_complete():
                # Wait for frames
                frames = self.pipeline.wait_for_frames()
                
                # Align frames
                aligned_frames = align.process(frames)
                aligned_depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                
                if not aligned_depth_frame or not color_frame:
                    continue
                
                # Convert to numpy arrays
                depth_image = np.asanyarray(aligned_depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                
                # Save frame data
                self.save_frame_data(color_image, depth_image, aligned_depth_frame)
                
                # Print progress every 5 seconds
                if self.frame_count % 150 == 0:  # Roughly every 5 seconds at 30fps
                    time_remaining = self.get_time_remaining()
                    print(f"Recording... {time_remaining:.1f}s remaining, {self.frame_count} frames captured")
                
                # Periodic memory cleanup for Jetson
                if self.frame_count % 150 == 0:
                    gc.collect()
            
            self.stop_recording_session()
                    
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
    parser.add_argument('--headless', action='store_true',
                       help='Run without GUI preview (headless mode)')
    
    args = parser.parse_args()
    
    # Validate duration
    if args.duration <= 0:
        print("Error: Duration must be greater than 0 seconds")
        return
    
    if args.duration > 3600:  # 1 hour limit
        print("Warning: Recording duration is longer than 1 hour")
        confirm = input("Continue? (y/n): ")
        if confirm.lower() != 'y':
            return
    
    print(f"Initializing Jetson timed recorder...")
    print(f"Duration: {args.duration} seconds ({args.duration/60:.1f} minutes)")
    print(f"Output folder: {args.output}")
    print(f"Mode: {'Headless' if args.headless else 'With preview'}")
    
    recorder = JetsonSimpleRecorder(data_folder=args.output, duration_seconds=args.duration)
    
    if args.headless:
        recorder.run_headless()
    else:
        recorder.run_with_preview()

if __name__ == "__main__":
    main()
