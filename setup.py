import os
import subprocess
import sys
from pathlib import Path

def create_venv():
    venv_path = "venv"
    
    # Remove existing venv if it exists
    if Path(venv_path).exists():
        print("Removing existing virtual environment...")
        import shutil
        shutil.rmtree(venv_path)
    
    print("Creating virtual environment...")
    try:
        subprocess.run([sys.executable, "-m", "venv", venv_path], check=True)
        
        # Get the correct pip and python paths
        if os.name == "nt":  # Windows
            python_path = os.path.join(venv_path, "Scripts", "python.exe")
            pip_path = os.path.join(venv_path, "Scripts", "pip.exe")
        else:  # Unix/Linux/Mac
            python_path = os.path.join(venv_path, "bin", "python")
            pip_path = os.path.join(venv_path, "bin", "pip")
            
        # Upgrade pip
        print("Upgrading pip...")
        subprocess.run([python_path, "-m", "pip", "install", "--upgrade", "pip"], check=True)
        
        # Install requirements
        print("Installing requirements...")
        subprocess.run([pip_path, "install", "-r", "requirements.txt"], check=True)
        
        print("\nSetup completed! You can activate the virtual environment using:")
        if os.name == "nt":
            print("venv\\Scripts\\activate")
        else:
            print("source venv/bin/activate")
            
    except subprocess.CalledProcessError as e:
        print(f"Error during setup: {e}")
        raise

if __name__ == "__main__":
    try:
        create_venv()
    except Exception as e:
        print(f"Error during setup: {str(e)}")
        sys.exit(1)

