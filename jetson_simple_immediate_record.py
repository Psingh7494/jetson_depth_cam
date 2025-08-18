#!/usr/bin/env python3
"""
Intel RealSense D435 Immediate Recording Script - Jetson AGX Xavier (Python 3.7)
Simple immediate recording that starts as soon as the script runs
Press CTRL+C or SPACE to stop recording
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import os
import json
import time
import argparse
import signal
import sys
import gc
from datetime import datetime, timedelta

class JetsonImmediateRecorder:
    def __init__(self, data_folder="data", duration_seconds=None):
        self.data_folder = data_folder
        self.duration_seconds = duration_seconds
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.recording = False
        self.session_folder = None
        self.frame_count = 0
        self.start_time = None
        self.end_time = None
        self.interrupted = False
        
        # Ensure data folder exists
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)
        
        # Set up signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # Configure streams
        self.setup_streams()
        
    def signal_handler(self, signum, frame):
        """Handle interrupt signals for graceful shutdown"""
        print(f"\nReceived interrupt signal. Stopping recording gracefully...")
        self.interrupted = True
        
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
            print("Error: RealSense camera with RGB sensor required")
            sys.exit(1)

        # Configure streams - optimized for Jetson AGX Xavier
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        if device_product_line == 'L500':
            self.config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
        else:
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            
    def start_recording_session(self):
        """Start immediate recording session"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_folder = os.path.join(self.data_folder, f"immediate_recording_{timestamp}")
        os.makedirs(self.session_folder, exist_ok=True)
        
        # Create subfolders for different data types
        os.makedirs(os.path.join(self.session_folder, "color"), exist_ok=True)
        os.makedirs(os.path.join(self.session_folder, "depth"), exist_ok=True)
        os.makedirs(os.path.join(self.session_folder, "depth_raw"), exist_ok=True)
        
        # Configure bag recording
        bag_filename = os.path.join(self.session_folder, f"immediate_recording_{timestamp}.bag")
        self.config.enable_record_to_file(bag_filename)
        
        self.recording = True
        self.frame_count = 0
        self.start_time = time.time()
        if self.duration_seconds:
            self.end_time = self.start_time + self.duration_seconds
        else:
            self.end_time = None  # Record indefinitely until interrupted
        
        # Create metadata file
        metadata = {
            "session_start": timestamp,
            "duration_seconds": self.duration_seconds if self.duration_seconds else "unlimited",
            "bag_file": f"immediate_recording_{timestamp}.bag",
            "platform": "Jetson AGX Xavier",
            "recording_type": "immediate",
            "format": {
                "color": "BGR8 640x480 @30fps",
                "depth": "Z16 640x480 @30fps"
            }
        }
        
        if self.duration_seconds:
            metadata["estimated_end_time"] = (datetime.now() + timedelta(seconds=self.duration_seconds)).strftime("%Y%m%d_%H%M%S")
        
        with open(os.path.join(self.session_folder, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=4)
            
        print(f"Immediate recording started: {self.session_folder}")
        if self.duration_seconds:
            print(f"Recording duration: {self.duration_seconds} seconds")
        else:
            print("Recording until interrupted (CTRL+C or SPACE)")
        
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
                metadata["stopped_by"] = "interrupt" if self.interrupted else "duration_completed"
                
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=4)
            
            print(f"\nRecording completed!")
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
        
        # Save frame metadata every 30 frames to reduce I/O
        if self.frame_count % 30 == 0:
            frame_metadata = {
                "frame_number": self.frame_count,
                "timestamp": datetime.now().isoformat(),
                "time_since_start": time.time() - self.start_time,
                "depth_scale": aligned_depth_frame.get_units()
            }
            
            frame_metadata_path = os.path.join(self.session_folder, f"metadata_frame_{self.frame_count:06d}.json")
            with open(frame_metadata_path, 'w') as f:
                json.dump(frame_metadata, f, indent=4)
            
        self.frame_count += 1
        
    def get_time_remaining(self):
        """Get remaining recording time in seconds"""
        if not self.recording or not self.start_time or not self.end_time:
            return None
        return max(0, self.end_time - time.time())
        
    def is_recording_complete(self):
        """Check if recording duration has been reached"""
        if not self.recording or not self.end_time:
            return False
        return time.time() >= self.end_time
        
    def run_with_preview(self):
        """Run immediate recording with simple preview"""
        try:
            # Start recording immediately
            self.start_recording_session()
            profile = self.pipeline.start(self.config)
            
            # Create alignment object
            align_to = rs.stream.color
            align = rs.align(align_to)
            
            print("\nRecording in progress...")
            if self.duration_seconds:
                print(f"Will record for {self.duration_seconds} seconds")
            print("Press SPACE to stop early or CTRL+C to quit")
            print("=" * 50)
            
            while self.recording and not self.interrupted:
                # Check if timed recording is complete
                if self.duration_seconds and self.is_recording_complete():
                    break
                
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
                
                # Create simple display
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                
                # Resize if needed
                if depth_colormap.shape[:2] != color_image.shape[:2]:
                    depth_colormap = cv2.resize(depth_colormap, (color_image.shape[1], color_image.shape[0]))
                
                # Create side-by-side display
                display_image = np.hstack((color_image, depth_colormap))
                
                # Add simple status text
                if self.duration_seconds:
                    time_remaining = self.get_time_remaining()
                    status_text = f"REC - {time_remaining:.1f}s left"
                else:
                    elapsed = time.time() - self.start_time
                    status_text = f"REC - {elapsed:.1f}s elapsed"
                
                cv2.putText(display_image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(display_image, f"Frames: {self.frame_count}", (10, display_image.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Display
                cv2.namedWindow('Jetson Immediate Recording', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('Jetson Immediate Recording', display_image)
                
                # Check for key press (non-blocking)
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):  # Spacebar to stop
                    print("Recording stopped by user (SPACE)")
                    break
                elif key == ord('q'):  # Q to quit
                    print("Recording stopped by user (Q)")
                    break
                
                # Print progress every 5 seconds
                if self.frame_count % 150 == 0:  # Roughly every 5 seconds at 30fps
                    if self.duration_seconds:
                        time_remaining = self.get_time_remaining()
                        print(f"Recording... {time_remaining:.1f}s remaining, {self.frame_count} frames captured")
                    else:
                        elapsed = time.time() - self.start_time
                        print(f"Recording... {elapsed:.1f}s elapsed, {self.frame_count} frames captured")
                
                # Periodic memory cleanup for Jetson
                if self.frame_count % 300 == 0:
                    gc.collect()
                    
        except Exception as e:
            print(f"Error during recording: {e}")
        finally:
            self.stop_recording_session()
            self.pipeline.stop()
            cv2.destroyAllWindows()
            print("Immediate recording session ended")
            
    def run_headless(self):
        """Run immediate recording without preview (headless mode)"""
        try:
            # Start recording immediately
            self.start_recording_session()
            profile = self.pipeline.start(self.config)
            
            # Create alignment object
            align_to = rs.stream.color
            align = rs.align(align_to)
            
            print("\nHeadless recording in progress...")
            if self.duration_seconds:
                print(f"Will record for {self.duration_seconds} seconds")
            else:
                print("Recording until interrupted (CTRL+C)")
            print("=" * 50)
            
            while self.recording and not self.interrupted:
                # Check if timed recording is complete
                if self.duration_seconds and self.is_recording_complete():
                    break
                
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
                    if self.duration_seconds:
                        time_remaining = self.get_time_remaining()
                        print(f"Recording... {time_remaining:.1f}s remaining, {self.frame_count} frames captured")
                    else:
                        elapsed = time.time() - self.start_time
                        print(f"Recording... {elapsed:.1f}s elapsed, {self.frame_count} frames captured")
                
                # Periodic memory cleanup for Jetson
                if self.frame_count % 150 == 0:
                    gc.collect()
            
            self.stop_recording_session()
                    
        except Exception as e:
            print(f"Error during recording: {e}")
        finally:
            self.stop_recording_session()
            self.pipeline.stop()
            print("Headless immediate recording completed")

def main():
    parser = argparse.ArgumentParser(description='Intel RealSense D435 Jetson Immediate Recording')
    parser.add_argument('-d', '--duration', type=int, default=None, 
                       help='Recording duration in seconds (default: unlimited)')
    parser.add_argument('-o', '--output', type=str, default='data',
                       help='Output folder for recordings (default: data)')
    parser.add_argument('--headless', action='store_true',
                       help='Run without GUI preview (headless mode)')
    
    args = parser.parse_args()
    
    # Validate duration
    if args.duration is not None and args.duration <= 0:
        print("Error: Duration must be greater than 0 seconds")
        return
    
    if args.duration and args.duration > 3600:  # 1 hour limit
        print("Warning: Recording duration is longer than 1 hour")
        confirm = input("Continue? (y/n): ")
        if confirm.lower() != 'y':
            return
    
    print(f"Jetson Immediate Recorder")
    print(f"========================")
    if args.duration:
        print(f"Duration: {args.duration} seconds ({args.duration/60:.1f} minutes)")
    else:
        print("Duration: Until interrupted (CTRL+C)")
    print(f"Output folder: {args.output}")
    print(f"Mode: {'Headless' if args.headless else 'With preview'}")
    print(f"Starting in 3 seconds...")
    
    # Give user a moment to read the info
    time.sleep(3)
    
    recorder = JetsonImmediateRecorder(data_folder=args.output, duration_seconds=args.duration)
    
    if args.headless:
        recorder.run_headless()
    else:
        recorder.run_with_preview()

if __name__ == "__main__":
    main()
