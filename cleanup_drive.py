import os
import shutil
import sys
import winreg
import psutil
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm
import tkinter as tk
from tkinter import ttk
import threading
from datetime import datetime, timedelta

console = Console()

class CleanupManager:
    def __init__(self):
        self.total_space_freed = 0
        self.paths_to_clean = []
        self.system_info = self.get_system_info()
        self.protected_paths = self.get_protected_paths()
        
    def get_protected_paths(self):
        """Get paths that should never be cleaned"""
        current_dir = os.getcwd()
        return [
            # Python environments
            os.path.join(current_dir, "venv"),
            os.path.join(current_dir, ".venv"),
            os.path.join(current_dir, ".env"),
            os.path.join(current_dir, "env"),
            
            # Project critical directories
            os.path.join(current_dir, "src"),
            os.path.join(current_dir, "tests"),
            os.path.join(current_dir, ".git"),
            
            # Active Python installation
            os.path.dirname(sys.executable),
            
            # Critical system paths
            os.environ.get('SystemRoot', 'C:\\Windows'),
            os.path.join(os.environ.get('SystemDrive', 'C:'), 'Program Files'),
            os.path.join(os.environ.get('SystemDrive', 'C:'), 'Program Files (x86)')
        ]
    
    def is_path_protected(self, path):
        """Check if a path should be protected from cleanup"""
        path = os.path.abspath(path)
        
        # Check if path is in protected paths
        for protected in self.protected_paths:
            if protected and (path.startswith(protected) or protected.startswith(path)):
                return True
        
        # Check if path contains Python environment indicators
        env_indicators = ['pyvenv.cfg', 'Scripts/python.exe', 'Scripts/activate']
        for indicator in env_indicators:
            if os.path.exists(os.path.join(path, indicator)):
                return True
                
        return False

    def get_size_str(self, size_bytes):
        """Convert bytes to human readable string"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.2f} TB"

    def get_safe_paths(self):
        """Get list of paths that can be safely cleaned"""
        home = os.path.expanduser("~")
        windows_temp = os.environ.get('TEMP')
        system_drive = os.environ.get('SystemDrive', 'C:')
        
        paths = [
            # Windows System Cleanable Files
            (windows_temp, "Windows Temp"),
            (f"{os.environ['SYSTEMROOT']}\\Temp", "System Temp"),
            (f"{system_drive}\\Windows\\SoftwareDistribution\\Download", "Windows Update Downloads"),
            (f"{system_drive}\\Windows\\Prefetch", "Windows Prefetch"),
            (f"{system_drive}\\Windows\\Installer\\$PatchCache$", "Windows Installer Cache"),
            (f"{home}\\AppData\\Local\\Temp", "User Temp Files"),
            
            # Browser Caches and Data
            (f"{home}\\AppData\\Local\\Google\\Chrome\\User Data\\Default\\Cache", "Chrome Cache"),
            (f"{home}\\AppData\\Local\\Google\\Chrome\\User Data\\Default\\Code Cache", "Chrome Code Cache"),
            (f"{home}\\AppData\\Local\\Google\\Chrome\\User Data\\Default\\GPUCache", "Chrome GPU Cache"),
            (f"{home}\\AppData\\Local\\Mozilla\\Firefox\\Profiles", "Firefox Cache"),
            (f"{home}\\AppData\\Local\\Microsoft\\Edge\\User Data\\Default\\Cache", "Edge Cache"),
            (f"{home}\\AppData\\Local\\Microsoft\\Edge\\User Data\\Default\\Code Cache", "Edge Code Cache"),
            (f"{home}\\AppData\\Local\\BraveSoftware\\Brave-Browser\\User Data\\Default\\Cache", "Brave Cache"),
            
            # Development Tools & IDEs
            (f"{home}\\.gradle\\caches", "Gradle Cache"),
            (f"{home}\\.m2\\repository", "Maven Cache"),
            (f"{home}\\AppData\\Local\\npm-cache", "NPM Cache"),
            (f"{home}\\AppData\\Local\\pip\\cache", "Pip Cache"),
            (f"{home}\\AppData\\Local\\Yarn\\Cache", "Yarn Cache"),
            (f"{home}\\AppData\\Local\\Composer\\Cache", "Composer Cache"),
            (f"{home}\\AppData\\Local\\NuGet\\Cache", "NuGet Cache"),
            (f"{home}\\.AndroidStudio", "Android Studio Cache"),
            (f"{home}\\.IntelliJIdea", "IntelliJ Cache"),
            (f"{home}\\.vscode", "VSCode Cache"),
            (f"{home}\\AppData\\Local\\JetBrains", "JetBrains Cache"),
            (f"{home}\\AppData\\Local\\Microsoft\\VisualStudio\\Packages", "Visual Studio Packages"),
            
            # ML Framework Caches
            (f"{home}\\.cache\\huggingface", "HuggingFace Cache"),
            (f"{home}\\.cache\\torch", "PyTorch Cache"),
            (f"{home}\\.cache\\pip", "Pip Cache"),
            (f"{home}\\AppData\\Local\\torch", "PyTorch Local Cache"),
            (f"{home}\\AppData\\Local\\huggingface", "HuggingFace Local Cache"),
            (f"{home}\\AppData\\Local\\pip", "Pip Local Cache"),
            
            # Container and VM Related
            (f"{home}\\.docker", "Docker Cache"),
            (f"{home}\\AppData\\Local\\Docker\\wsl", "Docker WSL Data"),
            (f"{home}\\AppData\\Local\\Packages\\CanonicalGroupLimited.Ubuntu*", "WSL Ubuntu Data"),
            (f"{home}\\.VirtualBox", "VirtualBox Cache"),
            (f"{home}\\AppData\\Local\\Temp\\VMware", "VMware Temp"),
            
            # Game Development & Unity
            (f"{home}\\AppData\\Local\\Unity", "Unity Cache"),
            (f"{home}\\AppData\\LocalLow\\Unity", "Unity Local Cache"),
            (f"{home}\\AppData\\Local\\Temp\\Unreal", "Unreal Engine Temp"),
            
            # Windows Store & Apps
            (f"{home}\\AppData\\Local\\Packages", "Windows Store Apps Cache"),
            (f"{home}\\AppData\\Local\\Microsoft\\Windows\\INetCache", "Internet Cache"),
            (f"{home}\\AppData\\Local\\Microsoft\\Windows\\Explorer", "Explorer Cache"),
            (f"{home}\\AppData\\Local\\Microsoft\\Windows\\WER", "Windows Error Reports"),
            
            # User Downloads and Temp
            (f"{home}\\Downloads", "Downloads Folder"),
            (f"{home}\\AppData\\Local\\Temp", "User Temp"),
            (f"{home}\\AppData\\LocalLow\\Temp", "User LocalLow Temp"),
            
            # Dataset and Model Directories
            ("./data", "Local Dataset Directory"),
            ("./datasets", "Local Datasets"),
            ("./model_testing", "Model Checkpoints"),
            ("./pretrained", "Pretrained Models"),
            ("./wandb", "Weights & Biases"),
            ("./lightning_logs", "PyTorch Lightning Logs"),
            ("./runs", "Training Runs"),
            ("./outputs", "Model Outputs"),
            ("./checkpoints", "Model Checkpoints"),
            
            # Log Files
            (f"{system_drive}\\Windows\\Logs", "Windows Logs"),
            (f"{system_drive}\\Windows\\Debug", "Windows Debug Logs"),
            ("./logs", "Application Logs"),
            ("./log", "Log Directory"),
            
            # Recycle Bin (requires admin)
            (f"{system_drive}\\$Recycle.Bin", "Recycle Bin"),
        ]
        
        # Add ML framework specific paths
        ml_frameworks = [
            "tensorflow", "keras", "pytorch", "transformers",
            "datasets", "models", "checkpoints", "wandb",
            "lightning_logs", "runs", "outputs", "results",
            "tensorboard", "optuna", "mlruns", "ray_results",
            "sacred", "neptune", "comet_ml", "mlflow"
        ]
        
        for framework in ml_frameworks:
            paths.append((f"{home}/.cache/{framework}", f"{framework.title()} Cache"))
            paths.append((f"{home}/AppData/Local/{framework}", f"{framework.title()} Local Cache"))
        
        # Add Visual Studio specific paths
        vs_versions = ["2017", "2019", "2022"]
        for version in vs_versions:
            paths.append((
                f"{home}\\AppData\\Local\\Microsoft\\VisualStudio\\{version}\\ComponentModelCache",
                f"VS {version} Component Cache"
            ))
        
        # Add Node.js specific paths
        paths.extend([
            (f"{home}\\AppData\\Roaming\\npm-cache", "NPM Cache (Roaming)"),
            (f"{home}\\AppData\\Roaming\\npm", "NPM Global Packages"),
            ("node_modules", "Node Modules"),
            (".next", "Next.js Build"),
            ("dist", "Distribution Build"),
            ("build", "Build Output"),
        ])
        
        # Add Python specific paths
        paths.extend([
            ("__pycache__", "Python Cache"),
            (".pytest_cache", "PyTest Cache"),
            (".mypy_cache", "MyPy Cache"),
            (".coverage", "Coverage Data"),
            ("htmlcov", "Coverage Report"),
            (".tox", "Tox Environment"),
            ("venv", "Virtual Environment"),
            (".env", "Virtual Environment"),
            (".venv", "Virtual Environment"),
        ])
        
        # Filter out Python virtual environments and critical paths
        filtered_paths = []
        for path, description in paths:
            if not self.is_path_protected(path):
                filtered_paths.append((path, description))
        
        return filtered_paths

    def analyze_path(self, path, description):
        """Analyze a path and return its size and file count"""
        try:
            if not os.path.exists(path):
                return None
            
            total_size = 0
            file_count = 0
            
            for root, dirs, files in os.walk(path):
                for file in files:
                    try:
                        file_path = os.path.join(root, file)
                        if os.path.exists(file_path):  # Check if file still exists
                            size = os.path.getsize(file_path)
                            total_size += size
                            file_count += 1
                    except (PermissionError, FileNotFoundError):
                        continue
            
            # Only include if has significant size (>1MB)
            if total_size > 1024 * 1024:
                last_modified = datetime.fromtimestamp(os.path.getmtime(path))
                days_old = (datetime.now() - last_modified).days
                
                return {
                    'path': path,
                    'description': description,
                    'size': total_size,
                    'size_str': self.get_size_str(total_size),
                    'files': file_count,
                    'days_old': days_old,
                    'last_modified': last_modified.strftime("%Y-%m-%d")
                }
        except Exception as e:
            console.print(f"[red]Error analyzing {path}: {str(e)}")
        
        return None

    def cleanup_selected_paths(self, selected_paths, progress_var=None, status_label=None):
        """Clean up selected paths with safety checks"""
        total_freed = 0
        
        for path_info in selected_paths:
            try:
                path = path_info['path']
                
                # Skip if path is protected
                if self.is_path_protected(path):
                    console.print(f"[yellow]Skipping protected path: {path}")
                    continue
                
                if os.path.exists(path):
                    size_before = path_info['size']
                    
                    if status_label:
                        status_label.config(text=f"Cleaning {path_info['description']}...")
                    
                    if os.path.isfile(path):
                        os.remove(path)
                    else:
                        # Use safe rmtree with error handling
                        def onerror(func, path, exc_info):
                            console.print(f"[yellow]Warning: Could not remove {path}")
                        
                        shutil.rmtree(path, onerror=onerror)
                    
                    total_freed += size_before
                    
                    if progress_var:
                        progress_var.set(progress_var.get() + 1)
            
            except Exception as e:
                console.print(f"[red]Error cleaning {path}: {str(e)}")
        
        return total_freed

    def restore_python_environment(self):
        """Attempt to restore Python environment if it was accidentally deleted"""
        try:
            if not os.path.exists("venv"):
                console.print("[yellow]Restoring Python virtual environment...")
                
                # Create new virtual environment
                subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
                
                # Determine the correct activation script
                if os.name == 'nt':  # Windows
                    activate_script = os.path.join("venv", "Scripts", "activate.bat")
                    pip_path = os.path.join("venv", "Scripts", "pip.exe")
                else:  # Unix-like
                    activate_script = os.path.join("venv", "bin", "activate")
                    pip_path = os.path.join("venv", "bin", "pip")
                
                # Install requirements if requirements.txt exists
                if os.path.exists("requirements.txt"):
                    subprocess.run([pip_path, "install", "-r", "requirements.txt"], check=True)
                
                console.print("[green]Python environment restored successfully!")
                console.print(f"[blue]Activate using: {activate_script}")
                
                return True
        except Exception as e:
            console.print(f"[red]Failed to restore Python environment: {str(e)}")
            return False

class CleanupGUI:
    def __init__(self):
        self.cleanup_manager = CleanupManager()
        self.root = tk.Tk()
        self.root.title("System Cleanup Tool")
        self.root.geometry("800x600")
        
        # Style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Treeview
        columns = ('Description', 'Size', 'Files', 'Last Modified', 'Age')
        self.tree = ttk.Treeview(main_frame, columns=columns, show='headings')
        
        # Configure columns
        self.tree.heading('Description', text='Description')
        self.tree.heading('Size', text='Size')
        self.tree.heading('Files', text='Files')
        self.tree.heading('Last Modified', text='Last Modified')
        self.tree.heading('Age', text='Days Old')
        
        # Column widths
        self.tree.column('Description', width=200)
        self.tree.column('Size', width=100)
        self.tree.column('Files', width=80)
        self.tree.column('Last Modified', width=100)
        self.tree.column('Age', width=80)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        # Progress bar and status
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100)
        self.status_label = ttk.Label(main_frame, text="Ready")
        
        # Buttons
        btn_frame = ttk.Frame(main_frame)
        self.scan_btn = ttk.Button(btn_frame, text="Scan System", command=self.scan_system)
        self.cleanup_btn = ttk.Button(btn_frame, text="Clean Selected", command=self.cleanup_selected)
        self.select_all_btn = ttk.Button(btn_frame, text="Select All", command=self.select_all)
        self.analyze_btn = ttk.Button(btn_frame, text="Analyze Drive", command=self.analyze_drive)
        self.programs_btn = ttk.Button(btn_frame, text="Check Programs", command=self.show_programs)
        
        # Layout
        self.tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        btn_frame.grid(row=1, column=0, columnspan=2, pady=10)
        self.scan_btn.pack(side=tk.LEFT, padx=5)
        self.select_all_btn.pack(side=tk.LEFT, padx=5)
        self.cleanup_btn.pack(side=tk.LEFT, padx=5)
        self.analyze_btn.pack(side=tk.LEFT, padx=5)
        self.programs_btn.pack(side=tk.LEFT, padx=5)
        
        self.progress.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        self.status_label.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        # Configure grid weights
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Store path info
        self.path_info_map = {}

    def scan_system(self):
        """Scan system for cleanable paths"""
        self.tree.delete(*self.tree.get_children())
        self.path_info_map.clear()
        self.status_label.config(text="Scanning system...")
        self.scan_btn.state(['disabled'])
        
        def scan():
            paths = self.cleanup_manager.get_safe_paths()
            results = []
            
            for path, description in paths:
                info = self.cleanup_manager.analyze_path(path, description)
                if info:
                    results.append(info)
            
            self.root.after(0, self.update_tree, results)
        
        thread = threading.Thread(target=scan)
        thread.start()

    def update_tree(self, results):
        """Update treeview with scan results"""
        for info in sorted(results, key=lambda x: x['size'], reverse=True):
            item = self.tree.insert('', 'end', values=(
                info['description'],
                info['size_str'],
                info['files'],
                info['last_modified'],
                info['days_old']
            ))
            self.path_info_map[item] = info
        
        self.scan_btn.state(['!disabled'])
        self.status_label.config(text="Scan complete")

    def select_all(self):
        """Select all items in treeview"""
        for item in self.tree.get_children():
            self.tree.selection_add(item)

    def cleanup_selected(self):
        """Clean up selected items"""
        selected = self.tree.selection()
        if not selected:
            self.status_label.config(text="No items selected")
            return
        
        selected_paths = [self.path_info_map[item] for item in selected]
        total_size = sum(path_info['size'] for path_info in selected_paths)
        
        if not Confirm.ask(
            f"Clean up {self.cleanup_manager.get_size_str(total_size)} from {len(selected)} locations?"):
            return
        
        self.progress_var.set(0)
        self.progress['maximum'] = len(selected)
        self.cleanup_btn.state(['disabled'])
        
        def cleanup():
            freed = self.cleanup_manager.cleanup_selected_paths(
                selected_paths, self.progress_var, self.status_label)
            
            self.root.after(0, self.cleanup_complete, freed)
        
        thread = threading.Thread(target=cleanup)
        thread.start()

    def cleanup_complete(self, freed):
        """Handle cleanup completion"""
        self.cleanup_btn.state(['!disabled'])
        self.status_label.config(
            text=f"Cleanup complete! Freed {self.cleanup_manager.get_size_str(freed)}")
        self.scan_system()  # Refresh the list

    def analyze_drive(self):
        """Perform comprehensive drive analysis"""
        self.status_label.config(text="Analyzing drive...")
        
        def analyze():
            # Get drive status
            cleanup_level = self.cleanup_manager.analyze_drive_space()
            
            # Get paths based on cleanup level
            paths = self.cleanup_manager.get_safe_paths()
            if cleanup_level in ["thorough", "aggressive"]:
                paths.extend(self.cleanup_manager.get_aggressive_cleanup_paths())
            
            # Find large files
            large_files = self.cleanup_manager.find_large_files()
            
            # Update UI with results
            self.root.after(0, self.update_analysis_results, paths, large_files)
        
        thread = threading.Thread(target=analyze)
        thread.start()
    
    def show_programs(self):
        """Show installed programs analysis"""
        programs = self.cleanup_manager.analyze_installed_programs()
        
        # Create new window
        program_window = tk.Toplevel(self.root)
        program_window.title("Installed Programs Analysis")
        program_window.geometry("600x400")
        
        # Create treeview
        columns = ('Name', 'Size', 'Install Date')
        tree = ttk.Treeview(program_window, columns=columns, show='headings')
        
        for col in columns:
            tree.heading(col, text=col)
        
        # Add programs sorted by size
        for prog in sorted(programs, key=lambda x: x['size'], reverse=True):
            tree.insert('', 'end', values=(
                prog['name'],
                self.cleanup_manager.get_size_str(prog['size']),
                prog['install_date']
            ))
        
        tree.pack(fill=tk.BOTH, expand=True)

    def run(self):
        """Start the GUI"""
        self.root.mainloop()

if __name__ == "__main__":
    cleanup_manager = CleanupManager()
    
    # Check if Python environment exists
    if not os.path.exists("venv/pyvenv.cfg"):
        console.print("[yellow]Python virtual environment not found!")
        if Confirm.ask("Would you like to restore the Python environment?"):
            if cleanup_manager.restore_python_environment():
                console.print("[green]Environment restored. Please restart the script.")
                sys.exit(0)
            else:
                console.print("[red]Failed to restore environment. Please recreate it manually.")
                sys.exit(1)
    
    # Check if running with admin privileges
    try:
        is_admin = os.getuid() == 0
    except AttributeError:
        import ctypes
        is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0

    if not is_admin:
        console.print("[red]Warning: Some cleanup operations may require administrator privileges")
        console.print("[yellow]Consider running this script as administrator for full functionality\n")
    
    app = CleanupGUI()
    app.run()

