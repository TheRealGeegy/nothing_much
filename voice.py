"""
Okada Voice Changer on Modal.com - PROPERLY FIXED
Based on Modal documentation at https://modal.com/docs

Key fixes:
1. Removed invalid .remote() call on web_server function
2. Fixed entrypoint to use proper Modal deployment patterns
3. Added --host 0.0.0.0 for proper binding
4. Added startup_timeout for slow initialization
5. Proper usage of modal serve vs modal deploy
"""

import modal
import subprocess
import os
import sys

APP_NAME = "okada-voice-changer"
GPU_TYPE = "T4"
CONTAINER_PORT = 18888
TIMEOUT_SECONDS = 3600
MEMORY_MB = 16384
STARTUP_TIMEOUT = 600  # 10 minutes for initial startup

VOLUME_NAME = "okada-vc-data"
MODELS_PATH = "/data"

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

app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


# APPROACH 1: Web Server (for production deployment)
# Deploy with: modal deploy docker_changer_fixed.py
# Or serve with: modal serve docker_changer_fixed.py
@app.function(
    gpu=GPU_TYPE,
    memory=MEMORY_MB,
    timeout=TIMEOUT_SECONDS,
    volumes={MODELS_PATH: volume},
    scaledown_window=300,
    concurrency_limit=1,            
    allow_concurrent_inputs=100, 
    keep_warm=0,            
)
@modal.web_server(CONTAINER_PORT, startup_timeout=STARTUP_TIMEOUT)
def voice_changer_server():
    """
    Web server for production deployment.
    
    Usage:
        modal deploy docker_changer_fixed.py  # For persistent deployment
        modal serve docker_changer_fixed.py   # For ephemeral testing
    
    Access via the URL that Modal provides (ends with .modal.run)
    
    IMPORTANT: This function starts the subprocess and returns immediately.
    Modal will detect when the server is ready and route traffic to it.
    """
    os.makedirs(f"{MODELS_PATH}/models", exist_ok=True)
    os.makedirs(f"{MODELS_PATH}/weights", exist_ok=True)
    
    print(f"Starting voice changer server on 0.0.0.0:{CONTAINER_PORT}")
    print(f"Model directory: {MODELS_PATH}/models")
    sys.stdout.flush()
    
    # Don't configure allowed-origins at all!
    # By default, the voice changer does NO CORS validation (see issue #1114)
    # This allows Modal's proxy to connect without any restrictions
    
    # Start the server process - it will keep running after this function returns
    # CRITICAL: Must bind to 0.0.0.0 for Modal
    cmd = [
        "python", "/voice-changer/server/MMVCServerSIO.py",
        "-p", str(CONTAINER_PORT),
        "--host", "0.0.0.0",  # REQUIRED for Modal
        "--https", "false",
        "--model_dir", f"{MODELS_PATH}/models",
        # NO --allowed-origins parameter = no CORS validation (default behavior)
    ]
    
    print(f"Using default CORS settings (no validation)")
    sys.stdout.flush()
    
    subprocess.Popen(
        cmd,
        cwd="/voice-changer/server",
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
    )
    
    # Function returns here, Modal will detect when server is ready on the port


# APPROACH 2: Tunnel (for temporary testing with modal run)
# Run with: modal run docker_changer_fixed.py::main --mode tunnel
@app.function(
    gpu=GPU_TYPE,
    memory=MEMORY_MB,
    timeout=TIMEOUT_SECONDS,
    volumes={MODELS_PATH: volume},
)
def voice_changer_tunnel():
    """
    Temporary tunnel for testing.
    
    Usage:
        modal run docker_changer_fixed.py::main --mode tunnel
    
    Creates a temporary URL that disappears when you press Ctrl+C
    """
    os.makedirs(f"{MODELS_PATH}/models", exist_ok=True)
    os.makedirs(f"{MODELS_PATH}/weights", exist_ok=True)
    
    with modal.forward(CONTAINER_PORT) as tunnel:
        print(f"\n{'='*60}")
        print(f"Voice Changer URL: {tunnel.url}")
        print(f"{'='*60}\n")
        sys.stdout.flush()
        
        # Build command without allowed-origins (uses default no CORS validation)
        cmd = [
            "python", "/voice-changer/server/MMVCServerSIO.py",
            "-p", str(CONTAINER_PORT),
            "--host", "0.0.0.0",  # REQUIRED for Modal
            "--https", "false",
            "--model_dir", f"{MODELS_PATH}/models",
            # NO --allowed-origins = no CORS validation (default)
        ]
        
        print("Using default CORS settings (no validation)")
        sys.stdout.flush()
        
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
                print(line.rstrip())
                sys.stdout.flush()
        except KeyboardInterrupt:
            print("\nShutting down...")
            proc.terminate()
            proc.wait()
        except Exception as e:
            print(f"Error: {e}")
            proc.terminate()
            proc.wait()
            raise


@app.function(volumes={MODELS_PATH: volume}, timeout=1800)
def download_pretrained_models():
    """Download required pretrained models"""
    import urllib.request
    
    weights_dir = f"{MODELS_PATH}/weights"
    os.makedirs(weights_dir, exist_ok=True)
    
    models = {
        "hubert_base.pt": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt",
        "rmvpe.pt": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt",
    }
    
    for filename, url in models.items():
        filepath = f"{weights_dir}/{filename}"
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            try:
                urllib.request.urlretrieve(url, filepath)
                print(f"✓ Downloaded {filename}")
            except Exception as e:
                print(f"✗ Failed to download {filename}: {e}")
    
    volume.commit()
    print("\n✓ All downloads complete!")


@app.function(volumes={MODELS_PATH: volume})
def list_models():
    """List all models in the data directory"""
    print(f"\nContents of {MODELS_PATH}:\n")
    for root, dirs, files in os.walk(MODELS_PATH):
        level = root.replace(MODELS_PATH, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        for file in files:
            size = os.path.getsize(os.path.join(root, file)) / (1024*1024)
            print(f'{" " * 2 * (level+1)}{file} ({size:.2f} MB)')


@app.local_entrypoint()
def main(mode: str = "tunnel", download_models: bool = False):
    """
    Main entrypoint for running the voice changer.
    
    Usage Examples:
        # Download models first
        modal run docker_changer_fixed.py::main --download-models
        
        # Run with temporary tunnel (for testing)
        modal run docker_changer_fixed.py::main --mode tunnel
        
        # List models
        modal run docker_changer_fixed.py::main --mode list
        
    For production deployment (persistent URL):
        modal deploy docker_changer_fixed.py
    
    For ephemeral deployment (testing, auto-updates):
        modal serve docker_changer_fixed.py
    """
    if download_models:
        print("Downloading pretrained models...")
        download_pretrained_models.remote()
    
    if mode == "tunnel":
        print("\n" + "="*70)
        print("Starting voice changer with temporary tunnel...")
        print("Press Ctrl+C to stop")
        print("="*70 + "\n")
        voice_changer_tunnel.remote()
    elif mode == "list":
        list_models.remote()
    else:
        print(f"Unknown mode: {mode}")
        print("\nAvailable modes:")
        print("  tunnel - Run with temporary tunnel (modal run)")
        print("  list   - List models in volume")
        print("\nFor production deployment, use:")
        print("  modal deploy docker_changer_fixed.py")
        print("\nFor ephemeral testing with auto-reload, use:")
        print("  modal serve docker_changer_fixed.py")