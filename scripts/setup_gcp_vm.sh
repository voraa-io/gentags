#!/bin/bash
# Setup script for Google Cloud VM
# Run this after SSH'ing into your VM

set -e

echo "=========================================="
echo "Gentags GCP VM Setup Script"
echo "=========================================="

# Check if running as root
if [ "$EUID" -eq 0 ]; then 
   echo "Please run as regular user (not root)"
   exit 1
fi

# 1. Update system
echo "1. Updating system packages..."
sudo apt update && sudo apt upgrade -y

# 2. Install dependencies
echo "2. Installing dependencies..."
sudo apt install -y python3 python3-pip python3-venv git screen curl build-essential

# 3. Install Poetry
echo "3. Installing Poetry..."
if ! command -v poetry &> /dev/null; then
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="$HOME/.local/bin:$PATH"
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
    source ~/.bashrc
    echo "Poetry installed"
    poetry --version
else
    echo "Poetry already installed"
    poetry --version
fi

# 4. Check NTP/time sync
echo "4. Checking NTP/time synchronization..."
if systemctl is-active --quiet systemd-timesyncd; then
    echo "✓ NTP is running"
else
    echo "⚠ Starting NTP service..."
    sudo systemctl enable systemd-timesyncd
    sudo systemctl start systemd-timesyncd
    echo "✓ NTP started"
fi

# 5. Check persistent disk mount
echo "5. Checking persistent disk mount..."
if mountpoint -q /mnt/results 2>/dev/null; then
    echo "✓ Persistent disk mounted at /mnt/results"
    mount | grep /mnt/results
    df -h /mnt/results
else
    echo "⚠ Persistent disk not mounted!"
    echo "  Finding disk..."
    lsblk -o NAME,SIZE,TYPE,MOUNTPOINT
    echo ""
    echo "  Please mount it manually:"
    echo "  # Find disk ID:"
    echo "  ls -l /dev/disk/by-id/ | grep gentags-results"
    echo "  # Format (first time only - ERASES DATA):"
    echo "  sudo mkfs.ext4 -F /dev/disk/by-id/google-gentags-results"
    echo "  # Mount:"
    echo "  sudo mkdir -p /mnt/results"
    echo "  sudo mount /dev/disk/by-id/google-gentags-results /mnt/results"
    echo "  # Add to fstab:"
    echo "  echo '/dev/disk/by-id/google-gentags-results /mnt/results ext4 defaults,nofail 0 2' | sudo tee -a /etc/fstab"
    echo "  sudo chown -R $USER:$USER /mnt/results"
    read -p "Press enter to continue anyway..."
fi

# 6. Create results directories
echo "6. Creating results directories..."
mkdir -p /mnt/results/results/raw
mkdir -p /mnt/results/results/meta
mkdir -p /mnt/results/results/logs
echo "✓ Directories created"

# 7. Check if repository exists
if [ -d "researchGentags" ]; then
    echo "7. Repository found, updating..."
    cd researchGentags
    git pull
else
    echo "7. Repository not found."
    echo "  Please clone it:"
    echo "  git clone https://github.com/your-username/researchGentags.git"
    echo "  cd researchGentags"
    read -p "Press enter when repository is cloned..."
    cd researchGentags
fi

# 8. Install project dependencies
echo "8. Installing project dependencies..."
poetry install
echo "✓ Dependencies installed"

# 9. Check .env file
echo "9. Checking .env file..."
if [ -f ".env" ]; then
    echo "✓ .env file exists"
else
    echo "⚠ .env file not found!"
    echo "  Please create it with your API keys:"
    echo "  nano .env"
    echo ""
    echo "  Add:"
    echo "  OPENAI_API_KEY=sk-..."
    echo "  GEMINI_API_KEY=..."
    echo "  ANTHROPIC_API_KEY=..."
    read -p "Press enter when .env is created..."
fi

# 10. Verify setup
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Verification:"
echo "  Python: $(python3 --version)"
echo "  Poetry: $(poetry --version)"
echo "  Git: $(git --version)"
echo "  Screen: $(screen -v)"
echo "  NTP: $(systemctl is-active systemd-timesyncd && echo 'running' || echo 'not running')"
echo "  Results dir: $(test -d /mnt/results/results && echo 'exists' || echo 'missing')"
echo ""
echo "Next steps:"
echo "  1. Verify API keys: poetry run python scripts/run_phase1.py --dry-run"
echo "  2. Run pilot: see docs/GCP_VM_SETUP.md"
echo "  3. Start shards: see docs/GCP_VM_SETUP.md"
echo ""

