"""
Okada Voice Changer - Using Pre-built Docker Image
Uses the official dannadori/vcclient image from Docker Hub.

Note: The official image is from August 2023 and may not have the latest features.
For newer versions, use voice_changer_simple.py which builds from source.

Usage:
    modal run voice_changer_docker.py         # Test interactively
    modal deploy voice_changer_docker.py      # Deploy permanently
"""

import modal
import subprocess
import os
import time

APP_NAME = "voice-changer-docker"
DOCKER_IMAGE = "dannadori/vcclient:20230826_211406"
PORT = 18888
GPU = "T4"

# The official image is based on NVIDIA's PyTorch image
# We need to add Python in a compatible way for Modal
image = (
    modal.Image.from_registry(
        DOCKER_IMAGE,
        add_python="3.10",  # Modal requires Python
        setup_dockerfile_commands=[
            # The image has its own entrypoint, we need to reset it
            "ENTRYPOINT []",
            "CMD []",
        ]
    )
    .apt_install("curl", "wget")  # Ensure we have basic utilities
)

app = modal.App(APP_NAME, image=image)
vol = modal.Volume.from_name("vc-docker-data", create_if_missing=True)

@app.function(
    gpu=GPU,
    memory=16384,
    timeout=3600,
    volumes={"/data": vol},
    container_idle_timeout=300,
)
@modal.web_server(PORT)
def server():
    """
    Start the voice changer server.
    The web UI will be available at the Modal-provided URL.
    """
    os.makedirs("/data/models", exist_ok=True)
    
    # The official Docker image structure
    # Server is at /voice-changer/server/MMVCServerSIO.py
    subprocess.Popen(
        [
            "python", "/voice-changer/server/MMVCServerSIO.py",
            "-p", str(PORT),
            "--https", "false",
            "--model_dir", "/data/models",
        ],
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "0",
        },
        cwd="/voice-changer/server"
    )
    
    print(f"Voice Changer starting on port {PORT}...")

@app.function(
    gpu=GPU,
    memory=16384,
    timeout=3600,
    volumes={"/data": vol},
)
def run_interactive():
    """Run with a tunnel for testing."""
    os.makedirs("/data/models", exist_ok=True)
    
    with modal.forward(PORT) as tunnel:
        print("\n" + "="*60)
        print("OKADA VOICE CHANGER (Docker Image)")
        print("="*60)
        print(f"URL: {tunnel.url}")
        print("="*60 + "\n")
        
        proc = subprocess.Popen(
            [
                "python", "/voice-changer/server/MMVCServerSIO.py",
                "-p", str(PORT),
                "--https", "false",
                "--model_dir", "/data/models",
            ],
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
            cwd="/voice-changer/server",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        
        try:
            while True:
                if proc.poll() is not None:
                    break
                line = proc.stdout.readline()
                if line:
                    print(line.decode(errors='replace').strip())
                time.sleep(0.01)
        except KeyboardInterrupt:
            print("\nStopping...")
            proc.terminate()

@app.local_entrypoint()
def main():
    """Run the voice changer interactively."""
    print("Starting voice changer with tunnel...")
    run_interactive.remote()