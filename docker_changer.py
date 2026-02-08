"""
Okada Voice Changer on Modal.com - FINAL FIX
Fixed: Directory removal error, CORS, Sample Models, Debugging

CRITICAL FIXES:
1. ✅ Uses shutil.rmtree() instead of os.rmdir() (handles non-empty dirs)
2. ✅ CORS: Uses actual Modal URLs (no wildcards)
3. ✅ Sample Models: Never pre-creates model_dir
4. ✅ Extensive debugging at every step
"""

import modal
import subprocess
import os
import sys
import shutil

APP_NAME = "okada-voice-changer"
GPU_TYPE = "T4"
CONTAINER_PORT = 18888
TIMEOUT_SECONDS = 3600
MEMORY_MB = 16384
STARTUP_TIMEOUT = 600

VOLUME_NAME = "okada-vc-data"
MODELS_PATH = "/data"

# Your Modal workspace name (change if different)
MODAL_WORKSPACE = "roarcrestapp"

# Create volume
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# Build image with dependencies
image = (
    modal.Image.from_registry(
        "nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04",
        add_python="3.10"
    )
    .apt_install(
        "git", "curl", "wget", "ffmpeg",
        "libsndfile1", "libsndfile1-dev",
        "portaudio19-dev", "libportaudio2",
        "build-essential", "cmake",
        "clang", "g++",
    )
    .pip_install("wheel", "setuptools", "cython", "pip")
    .pip_install(
        "torch==2.0.1+cu118",
        "torchaudio==2.0.2+cu118",
        extra_options="--extra-index-url https://download.pytorch.org/whl/cu118"
    )
    .run_commands("pip install numpy==1.24.3")
    .run_commands("pip install pyworld==0.3.4 --no-build-isolation")
    .run_commands(
        "CC=gcc CXX=g++ pip install git+https://github.com/facebookresearch/fairseq.git@v0.12.2 --no-build-isolation"
    )
    .pip_install(
        "scipy",
        "librosa==0.10.1",
        "soundfile",
        "praat-parselmouth",
        "torchcrepe",
        "faiss-cpu",
        "onnxruntime-gpu==1.16.3",
        "fastapi",
        "uvicorn[standard]",
        "python-multipart",
        "websockets",
        "python-socketio",
        "aiofiles",
        "httpx",
        "pydantic<2.0",
        "einops",
        "local-attention",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/w-okada/voice-changer.git /voice-changer"
    )
    .run_commands(
        "cd /voice-changer/server && pip install -r requirements.txt 2>/dev/null || true"
    )
)


def download_pretrained_models_to_volume():
    """Download HuBERT and RMVPE models during image build"""
    import urllib.request
    
    weights_dir = f"{MODELS_PATH}/weights"
    os.makedirs(weights_dir, exist_ok=True)
    
    models = {
        "hubert_base.pt": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt",
        "rmvpe.pt": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt",
    }
    
    print("=" * 80)
    print("IMAGE BUILD: Downloading pretrained models...")
    print("=" * 80)
    
    for filename, url in models.items():
        filepath = f"{weights_dir}/{filename}"
        if not os.path.exists(filepath):
            print(f"[BUILD] Downloading {filename}...")
            try:
                urllib.request.urlretrieve(url, filepath)
                size = os.path.getsize(filepath) / (1024*1024)
                print(f"[BUILD] ✓ Downloaded {filename} ({size:.1f} MB)")
            except Exception as e:
                print(f"[BUILD] ✗ Failed to download {filename}: {e}")
                raise
        else:
            size = os.path.getsize(filepath) / (1024*1024)
            print(f"[BUILD] ✓ {filename} already exists ({size:.1f} MB)")
    
    # CRITICAL: DO NOT CREATE models/ DIRECTORY!
    # The server checks if it exists to decide whether to download samples
    
    print("[BUILD] ✓ Pretrained models ready")
    print("[BUILD] ⚠ NOT creating models/ directory (let server handle it)")
    print("=" * 80)


image = image.run_function(
    download_pretrained_models_to_volume,
    volumes={MODELS_PATH: volume},
)

app = modal.App(APP_NAME, image=image)


@app.function(
    gpu=GPU_TYPE,
    memory=MEMORY_MB,
    timeout=TIMEOUT_SECONDS,
    volumes={MODELS_PATH: volume},
    scaledown_window=300,
    max_containers=5,
    min_containers=0,
)
@modal.concurrent(max_inputs=100)
@modal.web_server(CONTAINER_PORT, startup_timeout=STARTUP_TIMEOUT)
def voice_changer_server():
    """Production web server with proper CORS and debugging"""
    
    print("=" * 80)
    print("CONTAINER STARTUP: Okada Voice Changer Server")
    print("=" * 80)
    
    volume.reload()
    print("[STARTUP] ✓ Volume reloaded")
    
    weights_dir = f"{MODELS_PATH}/weights"
    models_dir = f"{MODELS_PATH}/models"
    
    # Verify pretrained models
    print(f"[STARTUP] Checking pretrained models in {weights_dir}...")
    hubert_path = f"{weights_dir}/hubert_base.pt"
    rmvpe_path = f"{weights_dir}/rmvpe.pt"
    
    if not os.path.exists(hubert_path):
        raise RuntimeError(f"[ERROR] hubert_base.pt not found at {hubert_path}")
    if not os.path.exists(rmvpe_path):
        raise RuntimeError(f"[ERROR] rmvpe.pt not found at {rmvpe_path}")
    
    hubert_size = os.path.getsize(hubert_path) / (1024*1024)
    rmvpe_size = os.path.getsize(rmvpe_path) / (1024*1024)
    print(f"[STARTUP] ✓ hubert_base.pt found ({hubert_size:.1f} MB)")
    print(f"[STARTUP] ✓ rmvpe.pt found ({rmvpe_size:.1f} MB)")
    
    # Check models directory - remove if exists to force fresh download
    if os.path.exists(models_dir):
        print(f"[STARTUP] ⚠ Models directory exists - checking contents...")
        try:
            items = os.listdir(models_dir)
            print(f"[STARTUP] Found {len(items)} items in models directory")
            
            # List what's there
            if len(items) > 0:
                print(f"[STARTUP] Contents:")
                for item in items[:5]:  # Show first 5
                    print(f"[STARTUP]   - {item}")
                if len(items) > 5:
                    print(f"[STARTUP]   ... and {len(items) - 5} more")
            
            # Always remove and let server recreate for clean state
            print(f"[STARTUP] Removing models directory to ensure clean download...")
            shutil.rmtree(models_dir)  # Use shutil.rmtree instead of os.rmdir
            print(f"[STARTUP] ✓ Models directory removed")
            
        except Exception as e:
            print(f"[STARTUP] ✗ Error removing models directory: {e}")
            raise
    else:
        print(f"[STARTUP] ✓ Models directory does NOT exist (server will download samples)")
    
    # CORS Configuration - Modal URL format
    function_name_normalized = "voice-changer-server"
    url_deploy = f"https://{MODAL_WORKSPACE}--{APP_NAME}-{function_name_normalized}.modal.run"
    url_serve = f"https://{MODAL_WORKSPACE}--{APP_NAME}-{function_name_normalized}-dev.modal.run"
    # SPACE-separated, not comma-separated!
    allowed_origins = f"http://localhost:{CONTAINER_PORT} {url_deploy} {url_serve}"
    
    print(f"[STARTUP] CORS allowed origins (space-separated):")
    print(f"  - http://localhost:{CONTAINER_PORT}")
    print(f"  - {url_deploy}")
    print(f"  - {url_serve}")
    
    # Create samples.json
    samples_json_path = "/tmp/samples.json"
    samples_json_content = '''{
    "sampleModels": "https://huggingface.co/wok000/vcclient_model/raw/main/samples_0004_o.json"
}'''
    
    with open(samples_json_path, 'w') as f:
        f.write(samples_json_content)
    print(f"[STARTUP] ✓ Created samples.json")
    
    # Build command
    cmd = [
        "python", "/voice-changer/server/MMVCServerSIO.py",
        "-p", str(CONTAINER_PORT),
        "--host", "0.0.0.0",
        "--https", "false",
        "--hubert_base", hubert_path,
        "--rmvpe", rmvpe_path,
        "--model_dir", models_dir,
        "--samples", samples_json_path,
        "--allowed-origins", allowed_origins,
    ]
    
    print("=" * 80)
    print("[STARTUP] Starting server with command:")
    print(f"  {' '.join(cmd)}")
    print("=" * 80)
    print()
    print("⏳ Server starting... Watch for sample model downloads below:")
    print()
    sys.stdout.flush()
    
    subprocess.Popen(
        cmd,
        cwd="/voice-changer/server",
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
    )


@app.function(
    gpu=GPU_TYPE,
    memory=MEMORY_MB,
    timeout=TIMEOUT_SECONDS,
    volumes={MODELS_PATH: volume},
    max_containers=1,
)
def voice_changer_tunnel():
    """Tunnel mode for testing"""
    
    print("=" * 80)
    print("TUNNEL MODE: Okada Voice Changer Server")
    print("=" * 80)
    
    volume.reload()
    print("[TUNNEL] ✓ Volume reloaded")
    
    with modal.forward(CONTAINER_PORT) as tunnel:
        print()
        print("=" * 80)
        print(f"🌐 Voice Changer URL: {tunnel.url}")
        print("=" * 80)
        print()
        
        weights_dir = f"{MODELS_PATH}/weights"
        models_dir = f"{MODELS_PATH}/models"
        
        # Check pretrained models
        print(f"[TUNNEL] Checking pretrained models...")
        hubert_path = f"{weights_dir}/hubert_base.pt"
        rmvpe_path = f"{weights_dir}/rmvpe.pt"
        
        if os.path.exists(hubert_path):
            size = os.path.getsize(hubert_path) / (1024*1024)
            print(f"[TUNNEL] ✓ hubert_base.pt ({size:.1f} MB)")
        else:
            print(f"[TUNNEL] ✗ hubert_base.pt NOT FOUND")
            
        if os.path.exists(rmvpe_path):
            size = os.path.getsize(rmvpe_path) / (1024*1024)
            print(f"[TUNNEL] ✓ rmvpe.pt ({size:.1f} MB)")
        else:
            print(f"[TUNNEL] ✗ rmvpe.pt NOT FOUND")
        
        # Check and clean models directory
        if os.path.exists(models_dir):
            print(f"[TUNNEL] ⚠ Models directory exists - checking contents...")
            try:
                items = os.listdir(models_dir)
                print(f"[TUNNEL] Found {len(items)} items")
                
                if len(items) > 0:
                    print(f"[TUNNEL] Contents:")
                    for item in items[:5]:
                        item_path = os.path.join(models_dir, item)
                        if os.path.isfile(item_path):
                            size = os.path.getsize(item_path) / (1024*1024)
                            print(f"[TUNNEL]   📄 {item} ({size:.1f} MB)")
                        else:
                            print(f"[TUNNEL]   📁 {item}/")
                    if len(items) > 5:
                        print(f"[TUNNEL]   ... and {len(items) - 5} more")
                
                print(f"[TUNNEL] Removing models directory for clean download...")
                shutil.rmtree(models_dir)  # Use shutil.rmtree - handles non-empty dirs!
                print(f"[TUNNEL] ✓ Models directory removed")
                
            except Exception as e:
                print(f"[TUNNEL] ✗ Error removing directory: {e}")
                raise
        else:
            print(f"[TUNNEL] ✓ Models directory does NOT exist (will download samples)")
        
        # CORS: Use the actual tunnel URL
        allowed_origins = tunnel.url
        print(f"[TUNNEL] CORS allowed origin: {allowed_origins}")
        
        # Create samples.json
        samples_json_path = "/tmp/samples.json"
        samples_json_content = '''{
    "sampleModels": "https://huggingface.co/wok000/vcclient_model/raw/main/samples_0004_o.json"
}'''
        
        with open(samples_json_path, 'w') as f:
            f.write(samples_json_content)
        print(f"[TUNNEL] ✓ Created samples.json")
        
        # Build command
        cmd = [
            "python", "/voice-changer/server/MMVCServerSIO.py",
            "-p", str(CONTAINER_PORT),
            "--host", "0.0.0.0",
            "--https", "false",
            "--hubert_base", hubert_path,
            "--rmvpe", rmvpe_path,
            "--model_dir", models_dir,
            "--samples", samples_json_path,
            "--allowed-origins", allowed_origins,
        ]
        
        print()
        print("=" * 80)
        print("[TUNNEL] Starting server...")
        print("=" * 80)
        print()
        print("📊 WATCH FOR THESE LOGS:")
        print("  ✓ [Voice Changer] download sample catalog. samples_0004_o.json")
        print("  ✓ [Voice Changer] Downloading sample model...")
        print("  ✓ GET /socket.io/ -> 200 OK")
        print()
        print("🚫 BAD SIGNS (if you see these, we have a problem):")
        print("  ✗ model_dir is already exists. skip download samples.")
        print("  ✗ POST /socket.io/ -> 400 Bad Request")
        print()
        print("=" * 80)
        print()
        sys.stdout.flush()
        
        # Start server with output
        proc = subprocess.Popen(
            cmd,
            cwd="/voice-changer/server",
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True
        )
        
        try:
            for line in proc.stdout:
                # Highlight important lines
                if "sample" in line.lower() and "download" in line.lower():
                    print(f"📥 {line.rstrip()}")
                elif "skip download" in line.lower():
                    print(f"⚠️  {line.rstrip()}")
                elif "400" in line or "bad request" in line.lower():
                    print(f"❌ {line.rstrip()}")
                elif "200 OK" in line:
                    print(f"✅ {line.rstrip()}")
                else:
                    print(line.rstrip())
                sys.stdout.flush()
        except KeyboardInterrupt:
            print("\n[TUNNEL] Shutting down...")
            proc.terminate()
            proc.wait()
        except Exception as e:
            print(f"[TUNNEL] Error: {e}")
            proc.terminate()
            proc.wait()
            raise


@app.function(volumes={MODELS_PATH: volume}, timeout=1800)
def force_download_models():
    """Re-download pretrained models"""
    download_pretrained_models_to_volume()
    volume.commit()
    print("\n✓ Pretrained models downloaded and committed!")


@app.function(volumes={MODELS_PATH: volume})
def clear_models_directory():
    """Clear models directory to force sample re-download"""
    volume.reload()
    models_dir = f"{MODELS_PATH}/models"
    
    print("=" * 80)
    print("CLEARING MODELS DIRECTORY")
    print("=" * 80)
    
    if os.path.exists(models_dir):
        try:
            items = os.listdir(models_dir)
            print(f"Found {len(items)} items in {models_dir}:")
            for item in items[:10]:  # Show first 10
                print(f"  - {item}")
            if len(items) > 10:
                print(f"  ... and {len(items) - 10} more")
            
            print(f"\nRemoving {models_dir}...")
            shutil.rmtree(models_dir)  # Use shutil.rmtree!
            volume.commit()
            print(f"✓ Models directory cleared!")
            print(f"✓ Sample models will be re-downloaded on next server start")
        except Exception as e:
            print(f"✗ Error clearing directory: {e}")
            raise
    else:
        print(f"Models directory doesn't exist - nothing to clear")
    
    print("=" * 80)


@app.function(volumes={MODELS_PATH: volume})
def list_volume():
    """List everything in the volume"""
    volume.reload()
    
    print("=" * 80)
    print(f"VOLUME CONTENTS: {MODELS_PATH}")
    print("=" * 80)
    print()
    
    for root, dirs, files in os.walk(MODELS_PATH):
        level = root.replace(MODELS_PATH, '').count(os.sep)
        indent = '  ' * level
        folder_name = os.path.basename(root) or MODELS_PATH
        print(f'{indent}📁 {folder_name}/')
        
        sub_indent = '  ' * (level + 1)
        for file in files:
            filepath = os.path.join(root, file)
            if os.path.exists(filepath):
                size = os.path.getsize(filepath) / (1024*1024)
                print(f'{sub_indent}📄 {file} ({size:.1f} MB)')
    
    print()
    print("=" * 80)


@app.function(volumes={MODELS_PATH: volume}, timeout=1800)
def upload_model(url: str, filename: str = ""):
    """Upload a custom voice model from URL"""
    import urllib.request
    
    volume.reload()
    models_dir = f"{MODELS_PATH}/models"
    os.makedirs(models_dir, exist_ok=True)
    
    if not filename:
        filename = url.split('/')[-1].split('?')[0]
        if not any(filename.endswith(ext) for ext in ['.pth', '.onnx', '.pt']):
            raise ValueError(f"Invalid filename: {filename}")
    
    output_path = os.path.join(models_dir, filename)
    
    print("=" * 80)
    print("UPLOADING CUSTOM MODEL")
    print("=" * 80)
    print(f"URL: {url}")
    print(f"Destination: {output_path}")
    print()
    
    def progress_hook(block_num, block_size, total_size):
        if total_size > 0:
            downloaded = block_num * block_size
            percent = min(downloaded * 100 / total_size, 100)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"\rDownloading: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='', flush=True)
    
    try:
        urllib.request.urlretrieve(url, output_path, reporthook=progress_hook)
        print("\n")
        volume.commit()
        
        size = os.path.getsize(output_path) / (1024*1024)
        print(f"✓ Model uploaded successfully!")
        print(f"  File: {filename}")
        print(f"  Size: {size:.1f} MB")
        print("=" * 80)
    except Exception as e:
        print(f"\n✗ Upload failed: {e}")
        raise


@app.local_entrypoint()
def main(mode: str = "tunnel", force_download: bool = False, clear_models: bool = False):
    """
    Main entrypoint
    
    Usage:
        # Test with tunnel
        modal run docker_changer.py::main --mode tunnel
        
        # Clear models directory
        modal run docker_changer.py::main --clear-models
        
        # List volume
        modal run docker_changer.py::main --mode list
        
        # Deploy to production
        modal deploy docker_changer.py
        
        # Serve with auto-reload
        modal serve docker_changer.py
    """
    
    if force_download:
        print("Force downloading pretrained models...")
        force_download_models.remote()
    
    if clear_models:
        print("Clearing models directory...")
        clear_models_directory.remote()
    
    if mode == "tunnel":
        print()
        print("=" * 80)
        print("🚀 STARTING OKADA VOICE CHANGER IN TUNNEL MODE")
        print("=" * 80)
        print()
        print("What to expect:")
        print("  1. Server will start")
        print("  2. Models directory will be cleaned (if exists)")
        print("  3. Sample models will be downloaded fresh")
        print("  4. You'll see sample models in the UI")
        print("  5. Socket.IO will connect successfully (no 400 errors)")
        print()
        print("Press Ctrl+C to stop")
        print()
        voice_changer_tunnel.remote()
    elif mode == "list":
        list_volume.remote()
    else:
        print(f"Unknown mode: {mode}")
        print("\nAvailable modes:")
        print("  tunnel - Run with tunnel")
        print("  list   - List volume contents")