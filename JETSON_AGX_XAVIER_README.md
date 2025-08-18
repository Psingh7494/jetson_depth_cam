# Intel RealSense D435 with NVIDIA Jetson AGX Xavier

This repository contains optimized Python scripts for using Intel RealSense D435 cameras with NVIDIA Jetson AGX Xavier development boards. The scripts leverage GPU acceleration and are specifically optimized for the Jetson platform.

## Hardware Requirements

- **NVIDIA Jetson AGX Xavier Development Kit**
- **Intel RealSense D435 Camera**
- **USB 3.0 Cable** (for connecting RealSense to Jetson)
- **Sufficient Storage** (for recorded data)
- **Cooling Solution** (recommended for extended recording sessions)

## Software Requirements

### JetPack Version
- **JetPack 4.6.1** or later (includes CUDA, cuDNN, TensorRT)
- **Ubuntu 18.04 LTS** (comes with JetPack)
- **Python 3.6+**

## Installation Guide

### 1. System Preparation

Update your Jetson system:
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-dev build-essential cmake git
```

### 2. Install Intel RealSense SDK

#### Method A: Using Pre-built Packages (Recommended)
```bash
# Add Intel's repository
sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE
sudo add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main"

# Install librealsense2
sudo apt update
sudo apt install -y librealsense2-dkms librealsense2-utils librealsense2-dev

# Verify installation
realsense-viewer
```

#### Method B: Build from Source (If pre-built packages don't work)
```bash
# Install dependencies
sudo apt install -y libssl-dev libusb-1.0-0-dev libudev-dev pkg-config libgtk-3-dev
sudo apt install -y libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev at

# Clone and build
git clone https://github.com/IntelRealSense/librealsense.git
cd librealsense
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON_BINDINGS=bool:true
make -j$(nproc)
sudo make install

# Update library paths
echo 'export PYTHONPATH=$PYTHONPATH:/usr/local/lib/python3.6/pyrealsense2' >> ~/.bashrc
source ~/.bashrc
```

### 3. Install Python Dependencies

Create a virtual environment (recommended):
```bash
python3 -m pip install --user virtualenv
python3 -m virtualenv ~/realsense_env
source ~/realsense_env/bin/activate
```

Install required packages:
```bash
# Core dependencies
pip install numpy>=1.19.0
pip install pyrealsense2>=2.54.1

# OpenCV with CUDA support (if not already installed)
pip install opencv-python>=4.5.0

# Additional utilities
pip install Pillow>=8.0.0
pip install matplotlib>=3.3.0
```

### 4. Install OpenCV with CUDA Support (Optional but Recommended)

For maximum performance, install OpenCV with CUDA support:

#### Option A: Pre-built OpenCV (Easier)
```bash
# Install pre-built OpenCV for Jetson
sudo apt install -y python3-opencv
```

#### Option B: Build OpenCV with CUDA (Better Performance)
```bash
# Remove any existing OpenCV installations
pip uninstall opencv-python opencv-contrib-python

# Install build dependencies
sudo apt install -y build-essential cmake git unzip pkg-config
sudo apt install -y libjpeg-dev libpng-dev libtiff-dev
sudo apt install -y libavcodec-dev libavformat-dev libswscale-dev
sudo apt install -y libgtk2.0-dev libcanberra-gtk-module libcanberra-gtk3-module
sudo apt install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
sudo apt install -y libxvidcore-dev libx264-dev libgtk-3-dev
sudo apt install -y libtbb2 libtbb-dev libdc1394-22-dev libv4l-dev
sudo apt install -y libopenblas-dev libatlas-base-dev libblas-dev
sudo apt install -y liblapack-dev gfortran libhdf5-dev
sudo apt install -y libprotobuf-dev libgoogle-glog-dev libgflags-dev
sudo apt install -y libgphoto2-dev libeigen3-dev libhdf5-dev doxygen

# Clone OpenCV
cd ~
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.5.0.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.5.0.zip
unzip opencv.zip
unzip opencv_contrib.zip

# Build OpenCV
cd opencv-4.5.0
mkdir build && cd build

cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib-4.5.0/modules \
      -D EIGEN_INCLUDE_PATH=/usr/include/eigen3 \
      -D WITH_OPENCL=OFF \
      -D WITH_CUDA=ON \
      -D CUDA_ARCH_BIN=7.2 \
      -D CUDA_ARCH_PTX="" \
      -D WITH_CUDNN=ON \
      -D WITH_CUBLAS=ON \
      -D ENABLE_FAST_MATH=ON \
      -D CUDA_FAST_MATH=ON \
      -D OPENCV_DNN_CUDA=ON \
      -D ENABLE_NEON=ON \
      -D WITH_QT=OFF \
      -D WITH_OPENMP=ON \
      -D BUILD_TIFF=ON \
      -D WITH_FFMPEG=ON \
      -D WITH_GSTREAMER=ON \
      -D WITH_TBB=ON \
      -D BUILD_TBB=ON \
      -D BUILD_TESTS=OFF \
      -D WITH_EIGEN=ON \
      -D WITH_V4L=ON \
      -D WITH_LIBV4L=ON \
      -D OPENCV_ENABLE_NONFREE=ON \
      -D INSTALL_C_EXAMPLES=OFF \
      -D INSTALL_PYTHON_EXAMPLES=OFF \
      -D PYTHON3_PACKAGES_PATH=/usr/lib/python3/dist-packages \
      -D OPENCV_GENERATE_PKGCONFIG=ON \
      -D BUILD_EXAMPLES=OFF ..

# Compile (this will take 1-2 hours)
make -j4
sudo make install
sudo ldconfig

# Verify CUDA support
python3 -c "import cv2; print('CUDA devices:', cv2.cuda.getCudaEnabledDeviceCount())"
```

### 5. Performance Optimizations

#### CPU Performance Mode
```bash
# Enable maximum performance mode
sudo nvpmodel -m 0
sudo jetson_clocks
```

#### Increase Swap Space (for building from source)
```bash
# Create swap file if needed
sudo fallocate -l 4G /var/swapfile
sudo chmod 600 /var/swapfile
sudo mkswap /var/swapfile
sudo swapon /var/swapfile

# Make permanent
echo '/var/swapfile swap swap defaults 0 0' | sudo tee -a /etc/fstab
```

## Python Libraries Summary

Create a `requirements_jetson.txt` file:
```txt
# Core RealSense support
pyrealsense2>=2.54.1

# Computer Vision (with CUDA support recommended)
opencv-python>=4.5.0

# Numerical computing
numpy>=1.19.0

# Image processing
Pillow>=8.0.0

# Visualization (optional)
matplotlib>=3.3.0

# System monitoring (optional)
psutil>=5.8.0

# Video encoding (optional)
imageio>=2.9.0
imageio-ffmpeg>=0.4.0
```

Install with:
```bash
pip install -r requirements_jetson.txt
```

## Usage Examples

### Live View (Jetson Optimized)
```bash
# Basic live view
python3 jetson_live_view.py

# High resolution with GPU acceleration
python3 jetson_live_view.py --preset high

# Ultra high quality (1280x720@60fps)
python3 jetson_live_view.py --preset ultra

# Disable GPU acceleration
python3 jetson_live_view.py --no-gpu
```

### Timed Recording (Jetson Optimized)
```bash
# Basic 30-second recording
python3 jetson_timed_record.py

# 5-minute HD recording with compression
python3 jetson_timed_record.py --preset hd --compress

# Headless recording (no GUI)
python3 jetson_timed_record.py --headless -d 300

# Ultra quality recording
python3 jetson_timed_record.py --preset ultra --compress
```

## Available Presets

### Live View Presets
- `low`: 640x480@30fps (good for basic testing)
- `medium`: 848x480@30fps (balanced quality/performance)
- `high`: 1280x720@30fps (high quality)
- `ultra`: 1280x720@60fps (maximum quality)

### Recording Presets
- `quick`: 10 seconds @ 640x480@30fps
- `short`: 30 seconds @ 640x480@30fps
- `medium`: 2 minutes @ 848x480@30fps
- `long`: 5 minutes @ 848x480@30fps
- `hd`: 2 minutes @ 1280x720@30fps
- `ultra`: 5 minutes @ 1280x720@60fps

## Performance Tips

### 1. Storage Optimization
- Use fast storage (NVMe SSD recommended)
- Enable compression for longer recordings: `--compress`
- Disable raw depth saving if not needed: `--no-raw-depth`

### 2. Memory Management
- Monitor system memory: `htop` or `watch -n 1 free -h`
- Use lower resolutions for very long recordings
- Enable swap space for memory-intensive operations

### 3. Thermal Management
- Monitor temperatures: `watch -n 1 tegrastats`
- Use active cooling for extended recording sessions
- Reduce frame rate if thermal throttling occurs

### 4. GPU Utilization
- Verify GPU acceleration: Look for "GPU: ON" in the display
- Monitor GPU usage: `nvidia-smi` or `tegrastats`
- Use `--no-gpu` flag if GPU causes issues

## Troubleshooting

### Common Issues

#### 1. Camera Not Detected
```bash
# Check USB connection
lsusb | grep Intel

# Check RealSense detection
rs-enumerate-devices

# Fix permissions
sudo usermod -a -G dialout $USER
# Logout and login again
```

#### 2. Low Frame Rate
- Check thermal throttling: `tegrastats`
- Enable performance mode: `sudo jetson_clocks`
- Reduce resolution or disable GPU acceleration
- Ensure sufficient power supply (use barrel jack, not USB-C)

#### 3. GPU Acceleration Not Working
```bash
# Check CUDA installation
nvcc --version

# Check OpenCV CUDA support
python3 -c "import cv2; print(cv2.getBuildInformation())" | grep -i cuda

# Verify GPU devices
python3 -c "import cv2; print('CUDA devices:', cv2.cuda.getCudaEnabledDeviceCount())"
```

#### 4. Out of Memory Errors
- Increase swap space
- Use lower resolution
- Enable compression: `--compress`
- Disable raw depth saving: `--no-raw-depth`

#### 5. Storage Full
```bash
# Check disk space
df -h

# Clean old recordings
rm -rf data/old_recordings_*

# Use external storage
python3 jetson_timed_record.py -o /media/external/recordings
```

## File Outputs

### Recording Structure
```
data/
└── jetson_recording_YYYYMMDD_HHMMSS/
    ├── jetson_recording_YYYYMMDD_HHMMSS.bag  # RealSense bag file
    ├── metadata.json                          # Recording metadata
    ├── color/                                 # Color images
    │   ├── frame_000000.png
    │   └── ...
    ├── depth/                                 # Depth colormaps
    │   ├── frame_000000.png
    │   └── ...
    ├── depth_raw/                            # Raw depth data (optional)
    │   ├── frame_000000.npy
    │   └── ...
    └── frame_XXXXXX_metadata.json           # Per-frame metadata
```

### Metadata Information
- Session start/end times
- Recording parameters (resolution, FPS, duration)
- Jetson-specific optimizations used
- Performance statistics (actual FPS, save efficiency)
- Hardware configuration

## Advanced Configuration

### Custom Resolutions
The scripts support these resolutions:
- 640x480 (Standard)
- 848x480 (Wide)
- 1280x720 (HD)

### Frame Rate Options
- 15 FPS (power saving)
- 30 FPS (standard)
- 60 FPS (high speed, HD only)

### GPU Memory Management
For extended recording sessions, you may need to adjust GPU memory:
```bash
# Check GPU memory
nvidia-smi

# Limit GPU memory usage in code (add to script)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
```

## Performance Benchmarks

### Expected Performance (Jetson AGX Xavier)

| Resolution | FPS | GPU | CPU Usage | Memory | Power |
|------------|-----|-----|-----------|--------|-------|
| 640x480    | 30  | ON  | ~25%      | ~2GB   | ~15W  |
| 848x480    | 30  | ON  | ~35%      | ~2.5GB | ~18W  |
| 1280x720   | 30  | ON  | ~45%      | ~3GB   | ~22W  |
| 1280x720   | 60  | ON  | ~65%      | ~3.5GB | ~28W  |

### Storage Requirements

| Resolution | FPS | Duration | Storage (Uncompressed) | Storage (Compressed) |
|------------|-----|----------|------------------------|----------------------|
| 640x480    | 30  | 1 minute | ~500MB                 | ~150MB               |
| 848x480    | 30  | 1 minute | ~750MB                 | ~200MB               |
| 1280x720   | 30  | 1 minute | ~1.5GB                 | ~400MB               |
| 1280x720   | 60  | 1 minute | ~3GB                   | ~800MB               |

## Support and Contributing

### Getting Help
1. Check this README for common issues
2. Verify hardware connections and power
3. Check Jetson system logs: `journalctl -f`
4. Monitor system resources: `tegrastats`

### Reporting Issues
Include the following information:
- JetPack version
- Hardware configuration (power mode, cooling)
- Script parameters used
- Error messages and logs
- System resource usage during error

### Contributing
Contributions are welcome! Please focus on:
- Jetson-specific optimizations
- Performance improvements
- Documentation updates
- Bug fixes

## License

This project is provided as-is for educational and research purposes. Please respect Intel RealSense SDK and NVIDIA software licenses.
