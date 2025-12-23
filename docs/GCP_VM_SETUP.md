# Google Cloud VM Setup Guide

This guide helps you set up and run the Week 2 Phase-1 extraction on Google Cloud Compute Engine.

## Prerequisites

- Google Cloud account with billing enabled
- `gcloud` CLI installed (or use browser SSH)
- Your repository pushed to GitHub (or uploaded to VM)

## Step 1: Create VM Instance

### Via Console:

1. Go to [Compute Engine > VM Instances](https://console.cloud.google.com/compute/instances)
2. Click "Create Instance"
3. **Machine configuration:**

   - Name: `gentags-extraction`
   - Machine type: `e2-medium` (2 vCPU, 4 GB RAM) - **recommended** for 3 parallel processes
   - Region: `us-central1` (or your preference)
   - Zone: `Any` (auto-select)

4. **Boot disk:**

   - OS: Ubuntu 22.04 LTS
   - Boot disk type: Balanced persistent disk
   - Size: 30 GB (minimum)

5. **Networking:** Leave defaults (external IP enabled)

6. **Firewall:** Leave defaults (no inbound ports needed for extraction-only)

   **Note:** If you plan to host a website later (e.g., for gentags viewer), you'll need to:

   - Open ports 80 (HTTP) and 443 (HTTPS) in firewall rules
   - Install and configure nginx
   - For now, keep defaults (no ports open) - this is correct for extraction-only

7. **Security:** Leave defaults (Shielded VM enabled)

8. Click "Create"

### Via gcloud CLI:

```bash
gcloud compute instances create gentags-extraction \
  --machine-type=e2-medium \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=20GB \
  --boot-disk-type=pd-balanced \
  --zone=us-central1-a
```

**Note:** `e2-medium` (2 vCPU, 4 GB RAM) is recommended over `e2-small` for better performance with 3 parallel Python processes, pandas operations, and logging.

## Step 2: Attach Persistent Disk for Results (IMPORTANT)

**Why:** VM local storage is ephemeral. Use a persistent disk for results.

### Create persistent disk:

```bash
gcloud compute disks create gentags-results \
  --size=50GB \
  --type=pd-balanced \
  --zone=us-central1-a
```

### Attach to VM:

```bash
gcloud compute instances attach-disk gentags-extraction \
  --disk=gentags-results \
  --zone=us-central1-a
```

### Format and mount (on VM):

```bash
# SSH into VM first
gcloud compute ssh gentags-extraction --zone=us-central1-a

# 1. Find the disk device name (don't assume /dev/sdb)
lsblk -o NAME,SIZE,TYPE,MOUNTPOINT
# Look for the disk without a mountpoint (should be ~50GB)

# 2. Find the disk by ID (more reliable)
ls -l /dev/disk/by-id/ | grep gentags-results
# Should show something like: google-gentags-results -> ../../sdb

# 3. Format disk (only first time - THIS ERASES DATA)
DISK_ID="/dev/disk/by-id/google-gentags-results"
sudo mkfs.ext4 -m 0 -F -E lazy_itable_init=0,lazy_journal_init=0,discard $DISK_ID

# 4. Create mount point
sudo mkdir -p /mnt/results

# 5. Mount disk
sudo mount -o discard,defaults $DISK_ID /mnt/results

# 6. Make it permanent (survives reboots) - using nofail so VM boots even if disk fails
echo "$DISK_ID /mnt/results ext4 defaults,nofail 0 2" | sudo tee -a /etc/fstab

# 7. Set permissions
sudo chown -R $USER:$USER /mnt/results

# 8. Verify mount
mount | grep /mnt/results
df -h /mnt/results
# Should show the disk mounted and ~50GB available
```

## Step 3: Setup Environment on VM

### SSH into VM:

```bash
gcloud compute ssh gentags-extraction --zone=us-central1-a
```

### Install dependencies:

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python, Poetry, Git, build tools
sudo apt install -y python3 python3-pip python3-venv git screen build-essential curl

# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Make Poetry PATH permanent (survives reconnects)
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Verify Poetry works
poetry --version

# Verify NTP is running (for correct timestamps)
sudo systemctl status systemd-timesyncd
# If not running: sudo systemctl enable systemd-timesyncd && sudo systemctl start systemd-timesyncd
```

### Clone repository:

```bash
cd ~
# Replace with your actual repository URL
git clone https://github.com/YOUR_USERNAME/researchGentags.git
# OR if private, use SSH:
# git clone git@github.com:YOUR_USERNAME/researchGentags.git
cd researchGentags
```

### Install project dependencies:

```bash
poetry install
```

### Setup API keys:

```bash
# Create .env file
nano .env
# Paste your API keys:
# OPENAI_API_KEY=sk-...
# GEMINI_API_KEY=...
# ANTHROPIC_API_KEY=...
```

### Verify persistent disk is mounted and create results directories:

```bash
# Verify disk is mounted
mount | grep /mnt/results
df -h /mnt/results
# Should show the persistent disk mounted with ~50GB available

# Create results directories
mkdir -p /mnt/results/results/raw
mkdir -p /mnt/results/results/meta
mkdir -p /mnt/results/results/logs
```

## Step 4: Run Extraction

### Option A: One pilot run, then 3 parallel shards (RECOMMENDED)

```bash
# 1. Run pilot once (pick one shard) - no screen needed, it's quick
cd ~/researchGentags
poetry run python scripts/run_phase1.py \
  --data data/study1_venues_20250117.csv \
  --models openai \
  --shard-name openai \
  --run-id week2_run_20251223_120000 \
  --total-max-cost-usd 50 --num-shards 3 \
  --pilot-venues 3 \
  --output /mnt/results/results \
  | tee /mnt/results/results/logs/pilot_openai.log

# Wait for pilot to complete (should finish in ~1-2 minutes)
# Check the log: tail -f /mnt/results/results/logs/pilot_openai.log

# 2. Once pilot completes, start 3 shards in parallel (use screen for these)
screen -S gentags_openai
poetry run python scripts/run_phase1.py \
  --data data/study1_venues_20250117.csv \
  --models openai \
  --shard-name openai \
  --run-id week2_run_20251223_120000 \
  --total-max-cost-usd 50 --num-shards 3 \
  --pilot-venues 0 \
  --output /mnt/results/results \
  | tee /mnt/results/results/logs/openai.log
# Ctrl+A, D to detach

screen -S gentags_gemini
poetry run python scripts/run_phase1.py \
  --data data/study1_venues_20250117.csv \
  --models gemini \
  --shard-name gemini \
  --run-id week2_run_20251223_120000 \
  --total-max-cost-usd 50 --num-shards 3 \
  --pilot-venues 0 \
  --output /mnt/results/results \
  | tee /mnt/results/results/logs/gemini.log
# Ctrl+A, D to detach

screen -S gentags_claude
poetry run python scripts/run_phase1.py \
  --data data/study1_venues_20250117.csv \
  --models claude \
  --shard-name claude \
  --run-id week2_run_20251223_120000 \
  --total-max-cost-usd 50 --num-shards 3 \
  --pilot-venues 0 \
  --output /mnt/results/results \
  | tee /mnt/results/results/logs/claude.log
# Ctrl+A, D to detach
```

### Option B: All shards run pilots (simpler but costs more)

```bash
# Just run the 3 shards - each will run its own pilot
# (The script will warn you, but you can proceed)
```

## Step 5: Monitor Progress

### Check running screens:

```bash
screen -ls
```

### Reattach to see progress:

```bash
screen -r gentags_openai  # or gentags_gemini, gentags_claude
# Ctrl+A, D to detach again
```

### Check logs:

```bash
tail -f /mnt/results/results/logs/openai.log
tail -f /mnt/results/results/logs/gemini.log
tail -f /mnt/results/results/logs/claude.log
```

### Check checkpoints:

```bash
ls -lh /mnt/results/results/week2_run_*_checkpoint_*.csv
```

### Check disk usage:

```bash
df -h /mnt/results
```

## Step 6: Download Results

### Option A: Download via gcloud:

```bash
# From your local machine
gcloud compute scp gentags-extraction:/mnt/results/results/week2_run_*_tags_*.csv ./results/ --zone=us-central1-a
gcloud compute scp gentags-extraction:/mnt/results/results/week2_run_*_extractions_*.csv ./results/ --zone=us-central1-a
gcloud compute scp gentags-extraction:/mnt/results/results/week2_run_*_cost_by_model_prompt_*.csv ./results/ --zone=us-central1-a
```

### Option B: Create snapshot and download:

```bash
# Create snapshot of persistent disk
gcloud compute disks snapshot gentags-results \
  --snapshot-names=gentags-results-final \
  --zone=us-central1-a

# Create new disk from snapshot (in your local region)
gcloud compute disks create gentags-results-copy \
  --source-snapshot=gentags-results-final \
  --zone=us-east1-a
```

## Firewall Rules (Future: Website Hosting)

**Current setup (extraction-only):** No inbound ports needed ✅

**If hosting website later:**

You'll need to open HTTP/HTTPS ports and set up nginx:

```bash
# Create firewall rule for HTTP
gcloud compute firewall-rules create allow-http \
  --allow tcp:80 \
  --source-ranges 0.0.0.0/0 \
  --target-tags http-server \
  --description "Allow HTTP traffic"

# Create firewall rule for HTTPS
gcloud compute firewall-rules create allow-https \
  --allow tcp:443 \
  --source-ranges 0.0.0.0/0 \
  --target-tags https-server \
  --description "Allow HTTPS traffic"

# Apply tags to VM
gcloud compute instances add-tags gentags-extraction \
  --tags http-server,https-server \
  --zone=us-central1-a
```

Then install nginx on the VM:

```bash
sudo apt install -y nginx
sudo systemctl enable nginx
sudo systemctl start nginx
```

**For now:** Keep defaults (no ports open) - this is correct for extraction-only.

## Troubleshooting

### VM disconnected / SSH timeout:

- Screens keep running in background
- Reconnect: `gcloud compute ssh gentags-extraction`
- Reattach: `screen -r gentags_openai`

### Check if processes are running:

```bash
ps aux | grep python
```

### Check disk space:

```bash
df -h
du -sh /mnt/results/results/*
```

### Resume from checkpoint:

```bash
# Just rerun the same command - it will auto-resume
poetry run python scripts/run_phase1.py \
  --models openai \
  --shard-name openai \
  --run-id week2_run_20251223_120000 \
  --output /mnt/results/results \
  ...
```

### Stop VM (but keep disk):

```bash
# From local machine
gcloud compute instances stop gentags-extraction --zone=us-central1-a

# To restart later:
gcloud compute instances start gentags-extraction --zone=us-central1-a
```

## Cost Estimates

- **VM:** e2-medium ~$0.06/hour = ~$4.32/day = ~$13 for 3 days
- **Persistent disk:** 50GB ~$0.17/month
- **API costs:** ~$14 total (from pilot estimates)
- **Total:** ~$27-30 for full run

## Best Practices

1. ✅ **Always use persistent disk** for results (not VM local storage)
2. ✅ **Use screen/tmux** so processes survive disconnects
3. ✅ **Log everything** with `tee` to persistent disk
4. ✅ **Check NTP** is running for accurate timestamps
5. ✅ **Monitor disk space** - results can be large
6. ✅ **Checkpoint every 50** extractions (default) - safe resume point
7. ✅ **Use same --run-id** for all shards to coordinate outputs

## Quick Reference

```bash
# Start screen session
screen -S name

# Detach: Ctrl+A, then D
# Reattach: screen -r name
# List: screen -ls
# Kill: screen -X -S name quit

# Check progress
tail -f /mnt/results/results/logs/*.log

# Check checkpoints
ls -lh /mnt/results/results/*checkpoint*.csv
```
