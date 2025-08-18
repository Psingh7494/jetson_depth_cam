#!/usr/bin/env python3
"""
Intel RealSense D435 Live View Script - Jetson AGX Xavier (Python 3.7)
Simple live view of RGB and depth streams optimized for Jetson AGX Xavier
Press 'q' to quit the application
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import sys
import gc

def main():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
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
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    try:
        pipeline.start(config)
        print("RealSense D435 Live View Started (Jetson AGX Xavier)")
        print("Press 'q' to quit")
        
        # Create alignment object to align depth frames to color frames
        align_to = rs.stream.color
        align = rs.align(align_to)
        
        # Frame counter for memory management
        frame_count = 0

        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            
            # Align the depth frame to color frame
            aligned_frames = align.process(frames)
            
            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not aligned_depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Apply colormap on depth image (optimized for Jetson)
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), 
                cv2.COLORMAP_JET
            )

            depth_colormap_dim = depth_colormap.shape
            color_colormap_dim = color_image.shape

            # If depth and color resolutions are different, resize color image to match depth image for display
            if depth_colormap_dim != color_colormap_dim:
                resized_color_image = cv2.resize(
                    color_image, 
                    dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), 
                    interpolation=cv2.INTER_AREA
                )
                images = np.hstack((resized_color_image, depth_colormap))
            else:
                images = np.hstack((color_image, depth_colormap))

            # Add simple text overlay
            cv2.putText(images, 'Color', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(images, 'Depth', (images.shape[1]//2 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Show images
            cv2.namedWindow('RealSense Jetson Live View', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense Jetson Live View', images)
            
            # Break loop with 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
            
            # Periodic memory cleanup for Jetson (every 10 seconds at 30fps)
            if frame_count % 300 == 0:
                gc.collect()

    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Stop streaming
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Live view stopped")

if __name__ == "__main__":
    main()
