import sys
import subprocess
import importlib.util

# --- Package Installation ---
REQUIRED_PACKAGES = [
    'PyQt5', 'torch', 'torchaudio', 'numpy', 'pyaudio', 
    'scipy', 'librosa', 'matplotlib', 'pystoi', 'pesq', 'soundfile'
]

def check_and_install_packages():
    """Checks and installs required packages."""
    print("Checking required packages...")
    for package in REQUIRED_PACKAGES:
        module_name = package
        if package == 'PyQt5': module_name = 'PyQt5'
        if package == 'pystoi': module_name = 'pystoi' # a.k.a. pystoi-rec
        if package == 'pesq': module_name = 'pesq'

        spec = importlib.util.find_spec(module_name)
        if spec is None:
            print(f"Package '{package}' not found. Attempting to install...")
            try:
                # Special case for pesq which needs a specific version for 'wb'
                if package == 'pesq':
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "pesq[speechmetrics]"])
                else:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"Successfully installed '{package}'.")
            except subprocess.CalledProcessError as e:
                print(f"ERROR: Failed to install '{package}'. Please install it manually. Error: {e}")
                sys.exit(1)
        else:
            print(f"{package} is already installed.")

# Run the package check
check_and_install_packages()

try:
    from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QTabWidget
    from training_tab import TrainingTab
    from streaming_tab import StreamingTab
    from evaluation_tab import EvaluationTab
except ImportError as e:
    print(f"Failed to import a required module: {e}")
    print("Please ensure all packages from REQUIRED_PACKAGES are installed.")
    sys.exit(1)


class MainWindow(QMainWindow):
    """The main application window which holds the tabbed interface."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Low-Latency Neural Audio Codec (< 20ms, 16kbps)")
        self.setGeometry(100, 100, 900, 700)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)

        self.training_tab = TrainingTab()
        self.streaming_tab = StreamingTab()
        self.evaluation_tab = EvaluationTab()

        self.tabs.addTab(self.training_tab, "Training")
        self.tabs.addTab(self.streaming_tab, "Real-Time Streaming (20ms)")
        self.tabs.addTab(self.evaluation_tab, "Model Evaluation")
        
    def closeEvent(self, event):
        """Ensures background threads are terminated when the application is closed."""
        print("Closing application...")
        self.streaming_tab.stop_streaming()
        self.training_tab.stop_training()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())

