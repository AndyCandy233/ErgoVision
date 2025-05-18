import sys
import subprocess
import os
from PyQt5 import QtWidgets, QtGui, QtCore

class Launcher(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Posture/Fatigue Monitoring Launcher")
        self.setWindowIcon(QtGui.QIcon.fromTheme("applications-system"))
        self.setFixedSize(420, 290)
        self.init_ui()
        self.sender_proc = None
        self.receiver_proc = None

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout()

        title = QtWidgets.QLabel("<h2>🧑‍💻 Posture & Fatigue Monitoring</h2>")
        title.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(title)

        grid = QtWidgets.QGridLayout()
        grid.setSpacing(10)

        # Sender options
        self.sender_btn = QtWidgets.QPushButton("Start Sender (Camera Sharing)")
        self.sender_btn.clicked.connect(self.start_sender)
        self.sender_stop_btn = QtWidgets.QPushButton("Stop Sender")
        self.sender_stop_btn.clicked.connect(self.stop_sender)
        self.sender_stop_btn.setEnabled(False)
        grid.addWidget(self.sender_btn, 0, 0)
        grid.addWidget(self.sender_stop_btn, 0, 1)

        # Receiver options
        self.receiver_btn = QtWidgets.QPushButton("Start Receiver (Dashboard)")
        self.receiver_btn.clicked.connect(self.start_receiver)
        self.receiver_stop_btn = QtWidgets.QPushButton("Stop Receiver")
        self.receiver_stop_btn.clicked.connect(self.stop_receiver)
        self.receiver_stop_btn.setEnabled(False)
        grid.addWidget(self.receiver_btn, 1, 0)
        grid.addWidget(self.receiver_stop_btn, 1, 1)

        # IP Option for Sender (dashboard host)
        self.ip_label = QtWidgets.QLabel("Dashboard Host IP for Sender:")
        self.ip_edit = QtWidgets.QLineEdit("10.37.123.168")
        grid.addWidget(self.ip_label, 2, 0)
        grid.addWidget(self.ip_edit, 2, 1)

        # Status
        self.status = QtWidgets.QLabel("<span style='color:gray'>Ready.</span>")
        grid.addWidget(self.status, 3, 0, 1, 2)

        layout.addLayout(grid)
        layout.addStretch()
        self.setLayout(layout)

    def start_sender(self):
        ip = self.ip_edit.text().strip()
        if not ip:
            self.status.setText("<span style='color:red'>Please enter a Dashboard Host IP.</span>")
            return
        # Optionally patch REPORT_HOST in sender script before running
        sender_path = os.path.join(os.getcwd(), "sender_with_stream.py")
        # Patch the IP in the sender script (very basic, for demo)
        with open(sender_path, "r") as f:
            src = f.read()
        new_src = src
        import re
        new_src = re.sub(r'(REPORT_HOST\s*=\s*")[^"]*(")', r'\1http://'+ip+r':8080/report\2', src)
        with open(sender_path, "w") as f:
            f.write(new_src)
        # Start sender
        self.sender_proc = subprocess.Popen([sys.executable, sender_path])
        self.sender_btn.setEnabled(False)
        self.sender_stop_btn.setEnabled(True)
        self.status.setText(f"<span style='color:blue'>Sender started. Streaming to {ip}:8080</span>")

    def stop_sender(self):
        if self.sender_proc:
            self.sender_proc.terminate()
            self.sender_proc.wait()
            self.sender_proc = None
            self.status.setText("<span style='color:green'>Sender stopped.</span>")
        self.sender_btn.setEnabled(True)
        self.sender_stop_btn.setEnabled(False)

    def start_receiver(self):
        receiver_path = os.path.join(os.getcwd(), "receiver.py")
        self.receiver_proc = subprocess.Popen(["python", receiver_path])
        self.receiver_btn.setEnabled(False)
        self.receiver_stop_btn.setEnabled(True)
        self.status.setText("<span style='color:blue'>Receiver (Dashboard) started at http://localhost:8080</span>")

    def stop_receiver(self):
        if self.receiver_proc:
            self.receiver_proc.terminate()
            self.receiver_proc.wait()
            self.receiver_proc = None
            self.status.setText("<span style='color:green'>Receiver stopped.</span>")
        self.receiver_btn.setEnabled(True)
        self.receiver_stop_btn.setEnabled(False)

    def closeEvent(self, event):
        # Cleanup on exit
        self.stop_sender()
        self.stop_receiver()
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    launcher = Launcher()
    launcher.show()
    sys.exit(app.exec_())