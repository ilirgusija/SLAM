#!/bin/bash
# Build script for CuPy v14 from GitHub source

set -e  # Exit on error

echo "=== Building CuPy v14 from GitHub ==="

# Step 1: Load CUDA module
echo "Loading CUDA module..."
module load cuda/12.2

# Step 2: Verify CUDA environment
echo "CUDA_PATH: $CUDA_PATH"
echo "nvcc version:"
nvcc --version

# Step 3: Set environment variables
export CUDA_PATH=${CUDA_PATH:-/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Core/cudacore/12.2.2}
export PATH=$CUDA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH

# Step 4: Clone CuPy repository (if not already cloned)
if [ ! -d "/tmp/cupy_build" ]; then
    echo "Cloning CuPy repository..."
    cd /tmp
    git clone --recursive https://github.com/cupy/cupy.git cupy_build
else
    echo "CuPy repository already exists, updating..."
    cd /tmp/cupy_build
    git pull
    git submodule update --init --recursive
fi

cd /tmp/cupy_build

# Step 5: Fix pyproject.toml for setuptools 68+ compatibility
echo "Fixing pyproject.toml for setuptools compatibility..."
# Fix license format: convert license = "MIT" to license = {text = "MIT"}
if grep -q 'license = "MIT"' pyproject.toml; then
    sed -i 's/license = "MIT"/license = {text = "MIT"}/' pyproject.toml
    echo "✓ Fixed license format"
fi

# Remove license-files if present (setuptools 68+ doesn't allow it in project section)
if grep -q 'license-files' pyproject.toml; then
    # Use Python to properly remove the license-files line from the [project] section
    python3 << 'EOF'
import re

with open('pyproject.toml', 'r') as f:
    content = f.read()

# Remove license-files line from [project] section
# Match [project] section and remove license-files line
lines = content.split('\n')
in_project = False
new_lines = []
for line in lines:
    if line.strip().startswith('[project]'):
        in_project = True
        new_lines.append(line)
    elif line.strip().startswith('[') and in_project:
        in_project = False
        new_lines.append(line)
    elif in_project and 'license-files' in line:
        # Skip this line
        continue
    else:
        new_lines.append(line)

with open('pyproject.toml', 'w') as f:
    f.write('\n'.join(new_lines))

print("✓ Removed license-files from [project] section")
EOF
fi

# Step 6: Install build dependencies
echo "Installing build dependencies..."
pip install Cython fastrlock numpy setuptools wheel

# Step 7: Build and install CuPy
echo "Building CuPy (this may take 10-30 minutes)..."
pip install --no-build-isolation .

echo "=== Build complete! ==="
echo "Verifying installation..."
python -c "import cupy as cp; print(f'CuPy version: {cp.__version__}')"
python -c "from cupyx.scipy.spatial import KDTree; print('✓ KDTree imported successfully!')"

