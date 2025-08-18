#!/bin/bash
# 
# Intel RealSense D435 Setup Script for NVIDIA Jetson AGX Xavier
# 
# This script automates the installation process for RealSense libraries
# and Python dependencies on Jetson AGX Xavier
#
# Usage: chmod +x setup_jetson.sh && ./setup_jetson.sh
#

set -e

echo "=================================================="
echo "Intel RealSense D435 Setup for Jetson AGX Xavier"
echo "=================================================="

# Check if running on Jetson
if [ ! -f /etc/nv_tegra_release ]; then
    echo "Warning: This script is designed for NVIDIA Jetson devices"
    echo "Continue anyway? (y/n)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Get Jetson info
echo "Jetson Device Information:"
cat /etc/nv_tegra_release
echo ""

# Update system
echo "Step 1: Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install basic dependencies
echo "Step 2: Installing basic dependencies..."
sudo apt install -y \
    python3-pip \
    python3-dev \
    python3-venv \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    htop \
    nvtop

# Install RealSense dependencies
echo "Step 3: Installing RealSense dependencies..."
sudo apt install -y \
    libssl-dev \
    libusb-1.0-0-dev \
    libudev-dev \
    pkg-config \
    libgtk-3-dev \
    libglfw3-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev

# Add Intel RealSense repository
echo "Step 4: Adding Intel RealSense repository..."
sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE || {
    echo "Failed to add Intel key via keyserver, trying alternative method..."
    wget -qO - https://librealsense.intel.com/Debian/apt-repo/conf/librealsense.gpg.key | sudo apt-key add -
}

sudo add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" -y

# Install RealSense libraries
echo "Step 5: Installing Intel RealSense libraries..."
sudo apt update
sudo apt install -y \
    librealsense2-dkms \
    librealsense2-utils \
    librealsense2-dev \
    librealsense2-dbg

# Create Python virtual environment
echo "Step 6: Creating Python virtual environment..."
python3 -m pip install --user virtualenv
python3 -m venv ~/realsense_jetson_env

# Activate virtual environment
source ~/realsense_jetson_env/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install Python dependencies
echo "Step 7: Installing Python dependencies..."
if [ -f "requirements_jetson.txt" ]; then
    pip install -r requirements_jetson.txt
else
    echo "Installing core dependencies..."
    pip install numpy>=1.19.0
    pip install pyrealsense2>=2.54.1
    pip install opencv-python>=4.5.0
    pip install Pillow>=8.0.0
    pip install psutil>=5.8.0
    pip install matplotlib>=3.3.0
fi

# Install jetson-stats for monitoring
echo "Step 8: Installing Jetson monitoring tools..."
sudo -H pip install jetson-stats

# Set up udev rules for RealSense
echo "Step 9: Setting up USB permissions..."
sudo usermod -a -G dialout $USER

# Check RealSense installation
echo "Step 10: Verifying RealSense installation..."
if command -v realsense-viewer &> /dev/null; then
    echo "✓ RealSense viewer installed successfully"
    echo "You can test the camera with: realsense-viewer"
else
    echo "✗ RealSense installation may have failed"
fi

# Check Python RealSense binding
echo "Step 11: Verifying Python bindings..."
if python3 -c "import pyrealsense2; print('✓ PyRealSense2 imported successfully')" 2>/dev/null; then
    echo "✓ Python RealSense bindings working"
else
    echo "✗ Python RealSense bindings not working"
fi

# Check OpenCV CUDA support
echo "Step 12: Checking OpenCV CUDA support..."
CUDA_DEVICES=$(python3 -c "import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())" 2>/dev/null || echo "0")
if [ "$CUDA_DEVICES" -gt 0 ]; then
    echo "✓ OpenCV CUDA support detected ($CUDA_DEVICES device(s))"
else
    echo "⚠ OpenCV CUDA support not available - performance will be limited"
    echo "Consider installing OpenCV with CUDA support for better performance"
fi

# Enable performance mode
echo "Step 13: Enabling performance mode..."
if command -v nvpmodel &> /dev/null; then
    sudo nvpmodel -m 0
    echo "✓ Performance mode enabled"
else
    echo "⚠ nvpmodel not found - performance mode not set"
fi

if command -v jetson_clocks &> /dev/null; then
    sudo jetson_clocks
    echo "✓ Jetson clocks maximized"
else
    echo "⚠ jetson_clocks not found - clocks not maximized"
fi

# Create activation script
echo "Step 14: Creating activation script..."
cat > ~/activate_realsense.sh << 'EOF'
#!/bin/bash
# Activate RealSense environment and set performance mode

echo "Activating RealSense Jetson environment..."

# Activate Python environment
source ~/realsense_jetson_env/bin/activate

# Set performance mode
sudo nvpmodel -m 0 2>/dev/null || echo "Could not set nvpmodel"
sudo jetson_clocks 2>/dev/null || echo "Could not set jetson_clocks"

echo "Environment activated!"
echo "Usage examples:"
echo "  python3 jetson_live_view.py --preset high"
echo "  python3 jetson_timed_record.py --preset hd"
echo ""
echo "Monitor performance with: tegrastats"
EOF

chmod +x ~/activate_realsense.sh

# Final instructions
echo ""
echo "=================================================="
echo "Installation Complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. Log out and log back in (for USB permissions)"
echo "2. Test the camera: realsense-viewer"
echo "3. Activate the environment: source ~/activate_realsense.sh"
echo "4. Run the optimized scripts:"
echo "   python3 jetson_live_view.py --preset high"
echo "   python3 jetson_timed_record.py --preset hd"
echo ""
echo "Monitor system performance with:"
echo "  tegrastats        # Overall system stats"
echo "  jtop              # Interactive Jetson monitor"
echo "  nvidia-smi        # GPU status"
echo ""
echo "Troubleshooting:"
echo "- If camera not detected: sudo usermod -a -G dialout \$USER (then logout/login)"
echo "- For performance issues: Check thermal throttling with tegrastats"
echo "- For memory issues: Monitor with htop or increase swap space"
echo ""

deactivate
