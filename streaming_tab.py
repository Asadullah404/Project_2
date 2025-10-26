# import os
# import threading
# from PyQt5.QtWidgets import (
#     QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit,
#     QLabel, QFileDialog, QTextEdit, QSpinBox, QComboBox, 
#     QGroupBox, QCheckBox, QListWidget
# )
# from PyQt5.QtCore import pyqtSignal, QObject, QThread, Qt
# from PyQt5.QtGui import QFont

# import socket
# import pyaudio
# import numpy as np
# import torch
# import time
# import librosa
# import struct

# from model import (
#     MuLawCodec, ALawCodec, DACCodec, EnCodecModel, TinyTransformerCodec, # Added TinyTransformerCodec
#     HOP_SIZE, DAC_AVAILABLE, ENCODEC_AVAILABLE
# )

# # --- Configuration ---
# BROADCAST_PORT = 37020
# STREAM_PORT = 37021
# DEVICE_ID = "NeuralCodecPC"
# FORMAT = pyaudio.paInt16
# CHANNELS = 1
# RATE = 16000
# CHUNK = HOP_SIZE  # 10ms (160 samples)

# # --- Network Discovery Worker (Unchanged) ---
# class DiscoveryWorker(QObject):
#     peer_found = pyqtSignal(str, str)
#     def __init__(self):
#         super().__init__()
#         self._running = True
        
#     def run(self):
#         with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
#             s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
#             try:
#                 s.bind(('', BROADCAST_PORT))
#             except OSError as e:
#                 print(f"Discovery: Could not bind to port {BROADCAST_PORT}. {e}")
#                 return
#             s.settimeout(1.0)
#             while self._running:
#                 try:
#                     data, addr = s.recvfrom(1024)
#                     message = data.decode()
#                     if message.startswith(DEVICE_ID):
#                         self.peer_found.emit(message.split(':')[1], addr[0])
#                 except socket.timeout:
#                     continue
#                 except Exception as e:
#                     print(f"Discovery error: {e}")
                    
#     def stop(self):
#         self._running = False

# # --- Audio Streaming Worker ---
# class StreamerWorker(QObject):
#     log_message = pyqtSignal(str)
    
#     def __init__(self, target_ip, model, model_type_str):
#         super().__init__()
#         self.target_ip = target_ip
#         self.model = model
#         self.model_type_str = model_type_str
#         self.p = pyaudio.PyAudio()
#         self._running = True
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.is_muted = False
#         self.file_playback_path = None
#         self.file_playback_event = threading.Event()
        
#         # Codec-specific buffering
#         self.send_buffer = []
#         self.recv_buffer = []
        
#         if isinstance(self.model, DACCodec):
#             self.chunk_size = 320  # DAC uses 20ms chunks
#         elif isinstance(self.model, EnCodecModel):
#             self.chunk_size = 480  # EnCodec may need different chunk size
#         elif isinstance(self.model, TinyTransformerCodec):
#              self.chunk_size = 160 # Use the minimum CHUNK size (10ms)
#         else:
#             self.chunk_size = CHUNK  # Default 10ms

#     def run(self):
#         self.sender_thread = threading.Thread(target=self.send_audio, daemon=True)
#         self.receiver_thread = threading.Thread(target=self.receive_audio, daemon=True)
#         self.sender_thread.start()
#         self.receiver_thread.start()

#     def set_mute(self, muted):
#         self.is_muted = muted
#         self.log_message.emit(f"Microphone {'muted' if muted else 'unmuted'}.")

#     def start_file_playback(self, filepath):
#         self.file_playback_path = filepath
#         self.file_playback_event.set()
#         self.log_message.emit(f"Queued file for playback: {filepath}")

#     def stop_file_playback(self):
#         self.file_playback_path = None
#         self.file_playback_event.clear()

#     def encode_data(self, data_bytes):
#         """Encodes raw audio bytes based on the selected model."""
#         try:
#             audio_np = np.frombuffer(data_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
#             if not self.model:
#                 return data_bytes
            
#             if isinstance(self.model, (MuLawCodec, ALawCodec)):
#                 audio_tensor = torch.from_numpy(audio_np).unsqueeze(0).unsqueeze(0)
#                 encoded_tensor = self.model.encode(audio_tensor)
#                 return encoded_tensor.cpu().numpy().tobytes()
            
#             # Grouping all neural codecs for chunked processing
#             if isinstance(self.model, (DACCodec, EnCodecModel, TinyTransformerCodec)):
#                 # Neural codec encoding with buffering
#                 self.send_buffer.extend(audio_np)
                
#                 # Check if we have enough data to meet the model's chunk requirement
#                 chunk_to_encode_size = self.chunk_size 
                
#                 if len(self.send_buffer) >= chunk_to_encode_size:
#                     chunk = np.array(self.send_buffer[:chunk_to_encode_size])
#                     self.send_buffer = self.send_buffer[chunk_to_encode_size:]
                    
#                     chunk_tensor = torch.from_numpy(chunk).unsqueeze(0).unsqueeze(0).to(self.device, dtype=torch.float32)
#                     with torch.no_grad():
#                         codes, orig_len = self.model.encode(chunk_tensor)
                        
#                         # Serialize codes for network transmission
#                         if isinstance(self.model, TinyTransformerCodec):
#                              # Custom model sends float32 (type 1)
#                             codes_np = codes.cpu().numpy().astype(np.float32)
#                             dtype_int = 1 
#                         elif isinstance(codes, torch.Tensor):
#                              # DAC/EnCodec tensor output sends int32 (type 0)
#                             codes_np = codes.cpu().numpy().astype(np.int32)
#                             dtype_int = 0 
#                         else:
#                             # EnCodec tuple/list format sends int32 (type 0)
#                             codes_np = codes[0].codes.cpu().numpy().astype(np.int32) if hasattr(codes[0], 'codes') else codes
#                             dtype_int = 0
                        
#                         # Pack with metadata (orig_len, num_dims, dtype_int, shape...)
#                         shape = codes_np.shape if hasattr(codes_np, 'shape') else (1,)
#                         header = struct.pack('I', orig_len) + struct.pack('I', len(shape)) + struct.pack('I', dtype_int)
#                         for s in shape:
#                             header += struct.pack('I', s)
                        
#                         payload = header + codes_np.tobytes()
#                         return payload
#                 else:
#                     return None # Not enough data to encode
#             else:
#                 return None # Not enough data for network chunk size

#         except Exception as e:
#             self.log_message.emit(f"Encoding error: {e}")
#             import traceback
#             self.log_message.emit(traceback.format_exc())
#             return None

#     def play_file_audio(self, filepath, s):
#         try:
#             wav, _ = librosa.load(filepath, sr=RATE, mono=True)
#             self.log_message.emit(f"Sending file: {filepath}...")
#             wav_int16 = (wav * 32767.0).astype(np.int16)
            
#             self.send_buffer = []
            
#             for i in range(0, len(wav_int16), CHUNK):
#                 if not self._running or self.file_playback_path != filepath:
#                     break
                
#                 chunk_data = wav_int16[i:i+CHUNK]
#                 if len(chunk_data) < CHUNK:
#                     chunk_data = np.pad(chunk_data, (0, CHUNK - len(chunk_data)), 'constant')
                
#                 chunk_bytes = chunk_data.tobytes()
#                 payload = self.encode_data(chunk_bytes)
                
#                 if payload:
#                     s.sendto(payload, (self.target_ip, STREAM_PORT))
                
#                 time.sleep(float(CHUNK) / RATE)
                
#             self.log_message.emit("File playback finished.")
#             self.send_buffer = []
            
#         except Exception as e:
#             self.log_message.emit(f"Error playing file: {e}")
#             self.stop_file_playback()

#     def send_audio(self):
#         try:
#             stream = self.p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
#             with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
#                 while self._running:
#                     if self.file_playback_event.is_set():
#                         self.file_playback_event.clear()
#                         if self.file_playback_path:
#                             self.play_file_audio(self.file_playback_path, s)
#                         if self.file_playback_path:
#                             self.file_playback_path = None
#                         continue
                    
#                     data = stream.read(CHUNK, exception_on_overflow=False)
#                     payload = self.encode_data(b'\x00' * (CHUNK * 2) if self.is_muted else data)
                    
#                     if payload:
#                         s.sendto(payload, (self.target_ip, STREAM_PORT))
                        
#         except Exception as e:
#             if self._running:
#                 self.log_message.emit(f"ERROR in send_audio: {e}")
#         finally:
#             if hasattr(self, 'p'):
#                 if 'stream' in locals() and stream.is_active():
#                     stream.stop_stream()
#                     stream.close()
#                 self.p.terminate()

#     def receive_audio(self):
#         try:
#             with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
#                 s.bind(('', STREAM_PORT))
#                 s.settimeout(2.0)
#                 stream = self.p.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True, frames_per_buffer=CHUNK)
                
#                 self.recv_buffer = []
                
#                 while self._running:
#                     try:
#                         data, _ = s.recvfrom(65536)
#                     except socket.timeout:
#                         if self._running:
#                             self.log_message.emit("Socket timeout, resetting decoder state.")
#                         self.recv_buffer = []
#                         continue
                        
#                     if not self.model:
#                         payload = data
#                     else:
#                         try:
#                             if isinstance(self.model, (MuLawCodec, ALawCodec)):
#                                 latent_tensor = torch.from_numpy(np.frombuffer(data, dtype=np.uint8)).unsqueeze(0)
#                                 decoded_tensor = self.model.decode(latent_tensor)
#                                 payload = (decoded_tensor.squeeze().cpu().numpy() * 32767.0).astype(np.int16).tobytes()
                                
#                             elif isinstance(self.model, (DACCodec, EnCodecModel, TinyTransformerCodec)): 
#                                 # Neural codec decoding
#                                 try:
#                                     # Unpack header: orig_len (I), num_dims (I), dtype (I)
#                                     offset = 0
#                                     orig_len = struct.unpack('I', data[offset:offset+4])[0]
#                                     offset += 4
                                    
#                                     num_dims = struct.unpack('I', data[offset:offset+4])[0]
#                                     offset += 4
                                    
#                                     dtype_int = struct.unpack('I', data[offset:offset+4])[0]
#                                     offset += 4
                                    
#                                     shape = []
#                                     for _ in range(num_dims):
#                                         shape.append(struct.unpack('I', data[offset:offset+4])[0])
#                                         offset += 4
                                    
#                                     codes_bytes = data[offset:]
                                    
#                                     # Determine numpy dtype based on header (1=float32, 0=int32)
#                                     dtype = np.float32 if dtype_int == 1 else np.int32
                                    
#                                     # Reconstruct codes tensor
#                                     codes_np = np.frombuffer(codes_bytes, dtype=dtype)
#                                     if shape:
#                                         codes_np = codes_np.reshape(shape)
                                    
#                                     codes = torch.from_numpy(codes_np).to(self.device)
                                    
#                                     with torch.no_grad():
#                                         decoded_tensor = self.model.decode(codes, orig_len)
                                    
#                                     decoded_audio = decoded_tensor.squeeze().cpu().numpy()
#                                     self.recv_buffer.extend(decoded_audio)
                                    
#                                     # Play back audio when enough data is buffered (using the default CHUNK)
#                                     while len(self.recv_buffer) >= CHUNK:
#                                         output_chunk = np.array(self.recv_buffer[:CHUNK])
#                                         self.recv_buffer = self.recv_buffer[CHUNK:]
#                                         payload = (output_chunk * 32767.0).astype(np.int16).tobytes()
#                                         stream.write(payload)
#                                         continue
                                    
#                                 except Exception as codec_err:
#                                     self.log_message.emit(f"Codec decode error: {codec_err}")
#                                     continue
#                         except Exception as e:
#                             self.log_message.emit(f"Decoding error: {e}")
#                             continue
                        
#                     stream.write(payload)
#         except Exception as e:
#             if self._running:
#                 self.log_message.emit(f"ERROR in receive_audio: {e}")
#         finally:
#             if hasattr(self, 'p'):
#                 if 'stream' in locals() and stream.is_active():
#                     stream.stop_stream()
#                     stream.close()
#                 self.p.terminate()

#     def stop(self):
#         self._running = False
#         self.stop_file_playback()


# class StreamingTab(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.peers = {}
#         self.streamer_worker = None
#         self.streamer_thread = None
#         self._setup_ui()
#         self._start_discovery()
#         self.model = None

#     def _setup_ui(self):
#         layout = QVBoxLayout(self)
#         layout.setSpacing(15)

#         config_group = QGroupBox("Configuration")
#         config_layout = QVBoxLayout(config_group)

#         model_layout = QHBoxLayout()
#         model_layout.addWidget(QLabel("<b>Codec:</b>"))
#         self.model_type_combo = QComboBox()
        
#         # All models listed here: Baselines, Custom, DAC, EnCodec
#         model_items = [
#             "Uncompressed",
#             "μ-Law Codec",  
#             "A-Law Codec",
#             "Tiny Transformer Codec (Custom, 16kHz, ~10ms)", 
#         ]
#         if DAC_AVAILABLE:
#             model_items.append("DAC Codec (16kHz, 20ms)")
#         if ENCODEC_AVAILABLE:
#             model_items.append("EnCodec (24kHz, 6kbps)")
        
#         self.model_type_combo.addItems(model_items)
#         self.model_type_combo.currentTextChanged.connect(self.on_model_type_changed)
#         model_layout.addWidget(self.model_type_combo)
#         config_layout.addLayout(model_layout)
        
#         self.model_path_edit = QLineEdit()
#         self.model_path_edit.setPlaceholderText("Path to TRAINED model (.pth). Mandatory for Tiny Transformer Codec.") 
#         self.browse_model_button = QPushButton("Browse...")
#         self.browse_model_button.clicked.connect(self.browse_model)
#         self.load_model_button = QPushButton("Load Model")
#         self.load_model_button.clicked.connect(self.load_model)
#         model_path_layout = QHBoxLayout()
#         model_path_layout.addWidget(self.model_path_edit)
#         model_path_layout.addWidget(self.browse_model_button)
#         model_path_layout.addWidget(self.load_model_button)
#         config_layout.addLayout(model_path_layout)
        
#         self.bitrate_label = QLabel(f"Codec Info: Select a codec")
#         config_layout.addWidget(self.bitrate_label)

#         self.play_file_path_edit = QLineEdit()
#         self.play_file_path_edit.setPlaceholderText("Path to audio file for playback...")
#         self.browse_play_file_button = QPushButton("Browse...")
#         self.browse_play_file_button.clicked.connect(self.browse_play_file)
#         file_playback_layout = QHBoxLayout()
#         file_playback_layout.addWidget(self.play_file_path_edit)
#         file_playback_layout.addWidget(self.browse_play_file_button)
#         config_layout.addLayout(file_playback_layout)
#         layout.addWidget(config_group)

#         network_group = QGroupBox("Network")
#         network_layout = QVBoxLayout(network_group)
#         network_layout.addWidget(QLabel("<b>Available Peers:</b>"))
#         self.peer_list = QListWidget()
#         self.peer_list.setMaximumHeight(100)
#         self.refresh_button = QPushButton("Refresh List")
#         self.refresh_button.clicked.connect(self.send_broadcast)
#         network_layout.addWidget(self.peer_list)
#         network_layout.addWidget(self.refresh_button, Qt.AlignRight)
#         layout.addWidget(network_group)

#         controls_group = QGroupBox("Controls")
#         controls_layout = QHBoxLayout(controls_group)
#         self.connect_button = QPushButton("Start Streaming")
#         self.connect_button.setStyleSheet("background-color: #4CAF50; color: white; padding: 5px; border-radius: 5px;")
#         self.connect_button.clicked.connect(self.start_streaming)
#         self.disconnect_button = QPushButton("Stop Streaming")
#         self.disconnect_button.setStyleSheet("background-color: #f44336; color: white; padding: 5px; border-radius: 5px;")
#         self.disconnect_button.setEnabled(False)
#         self.disconnect_button.clicked.connect(self.stop_streaming)
#         self.play_file_button = QPushButton("▶ Play File")
#         self.play_file_button.setEnabled(False)
#         self.play_file_button.clicked.connect(self.start_file_playback)
#         self.mute_mic_checkbox = QCheckBox("Mute Mic")
#         self.mute_mic_checkbox.stateChanged.connect(self.on_mute_changed)
#         self.status_label = QLabel("<b>Status:</b> <font color='red'>Disconnected</font>")
        
#         controls_layout.addWidget(self.connect_button)
#         controls_layout.addWidget(self.disconnect_button)
#         controls_layout.addWidget(self.play_file_button)
#         controls_layout.addStretch()
#         controls_layout.addWidget(self.mute_mic_checkbox)
#         controls_layout.addWidget(self.status_label)
#         layout.addWidget(controls_group)

#         log_group = QGroupBox("Log")
#         log_layout = QVBoxLayout(log_group)
#         self.log_text_edit = QTextEdit()
#         self.log_text_edit.setReadOnly(True)
#         log_layout.addWidget(self.log_text_edit)
#         layout.addWidget(log_group)
        
#         self.on_model_type_changed(self.model_type_combo.currentText())

#     def log(self, message):
#         self.log_text_edit.append(message)
#         self.log_text_edit.verticalScrollBar().setValue(self.log_text_edit.verticalScrollBar().maximum())

#     def on_model_type_changed(self, model_name):
#         is_neural = any(x in model_name for x in ["DAC", "EnCodec", "Transformer"])
#         is_custom_transformer = "Tiny Transformer Codec" in model_name
        
#         self.model_path_edit.setEnabled(is_neural)
#         self.browse_model_button.setEnabled(is_neural)
#         self.load_model_button.setEnabled(is_neural)

#         if not is_neural:
#             self.model_path_edit.setText("N/A (Traditional or Uncompressed)")
#             bitrate = '256 kbps' if 'Uncompressed' in model_name else '128 kbps'
#             self.bitrate_label.setText(f"Codec Bitrate: {bitrate}")
#             self.load_model()
#         elif "DAC" in model_name:
#             self.model_path_edit.setText("(auto-download if not provided)")
#             self.bitrate_label.setText(f"Codec: DAC | Bitrate: ~8-12 kbps | Latency: 20ms")
#         elif "EnCodec" in model_name:
#             self.model_path_edit.setText("(auto-download if not provided)")
#             self.bitrate_label.setText(f"Codec: EnCodec | Bitrate: 6 kbps | Latency: ~10ms")
#         elif is_custom_transformer:
#              self.model_path_edit.setText("MANDATORY: Path to your trained checkpoint file (.pth)")
#              self.bitrate_label.setText(f"Codec: Tiny Transformer | Bitrate: ~8 kbps (Estimated) | Latency: ~10ms")

#     def on_mute_changed(self, state):
#         if self.streamer_worker:
#             self.streamer_worker.set_mute(state == Qt.Checked)

#     def browse_play_file(self):
#         filepath, _ = QFileDialog.getOpenFileName(self, "Select Audio File", "", "Audio Files (*.wav *.mp3)")
#         if filepath:
#             self.play_file_path_edit.setText(filepath)

#     def start_file_playback(self):
#         if not self.play_file_path_edit.text():
#             self.log("ERROR: Please select a file to play.")
#             return
#         if self.streamer_worker:
#             self.streamer_worker.stop_file_playback()
#             self.streamer_worker.start_file_playback(self.play_file_path_edit.text())
#         else:
#             self.log("ERROR: Must be streaming to play a file.")

#     def browse_model(self):
#         filepath, _ = QFileDialog.getOpenFileName(self, "Select Model File", "", "PyTorch Models (*.pth *.pt)")
#         if filepath:
#             self.model_path_edit.setText(filepath)

#     def load_model(self):
#         model_type_str = self.model_type_combo.currentText()
#         self.model = None

#         if "Uncompressed" in model_type_str:
#             self.model = None
#             self.log("Streaming uncompressed (256 kbps).")
#             return True
#         elif "μ-Law" in model_type_str:
#             self.model = MuLawCodec()
#             self.log("Loaded μ-Law Codec (128 kbps).")
#             return True
#         elif "A-Law" in model_type_str:
#             self.model = ALawCodec()
#             self.log("Loaded A-Law Codec (128 kbps).")
#             return True
#         elif "DAC" in model_type_str:
#             if not DAC_AVAILABLE:
#                 self.log("ERROR: DAC not installed. Install with: pip install descript-audio-codec")
#                 return False
#             try:
#                 path = self.model_path_edit.text()
#                 if "auto-download" in path or not path or "N/A" in path:
#                     path = None
#                 self.model = DACCodec(model_path=path, model_type="16khz")
#                 self.log("Loaded DAC Codec.")
#                 return True
#             except Exception as e:
#                 self.log(f"ERROR: Failed to load DAC: {e}")
#                 return False
#         elif "EnCodec" in model_type_str:
#             if not ENCODEC_AVAILABLE:
#                 self.log("ERROR: EnCodec not installed. Install with: pip install encodec")
#                 return False
#             try:
#                 path = self.model_path_edit.text()
#                 if "auto-download" in path or not path or "N/A" in path:
#                     path = None
#                 self.model = EnCodecModel(model_path=path)
#                 self.log("Loaded EnCodec.")
#                 return True
#             except Exception as e:
#                 self.log(f"ERROR: Failed to load EnCodec: {e}")
#                 return False
#         elif "Tiny Transformer Codec" in model_type_str: # Custom Codec Loading
#             model_path = self.model_path_edit.text()
#             if "MANDATORY" in model_path or not os.path.exists(model_path):
#                 self.log("ERROR: Please provide a valid path to your trained Tiny Transformer Codec (.pth) model.")
#                 return False
#             try:
#                 self.model = TinyTransformerCodec.load_model(model_path)
#                 self.log("Loaded Tiny Transformer Codec.")
#                 return True
#             except Exception as e:
#                 self.log(f"ERROR: Failed to load Tiny Transformer Codec: {e}")
#                 return False
            
#         return False

#     def add_peer(self, name, ip):
#         if ip not in self.peers.values():
#             self.peers[name] = ip
#             self.peer_list.addItem(f"{name} ({ip})")

#     def send_broadcast(self):
#         try:
#             with socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP) as s:
#                 s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
#                 s.sendto(f"{DEVICE_ID}:{socket.gethostname()}".encode(), ('<broadcast>', BROADCAST_PORT))
#             self.log("Sent discovery broadcast.")
#         except Exception as e:
#             self.log(f"Error sending broadcast: {e}")

#     def _start_discovery(self):
#         self.discovery_thread = QThread()
#         self.discovery_worker = DiscoveryWorker()
#         self.discovery_worker.moveToThread(self.discovery_thread)
#         self.discovery_worker.peer_found.connect(self.add_peer)
#         self.discovery_thread.started.connect(self.discovery_worker.run)
#         self.discovery_thread.start()
#         self.send_broadcast()

#     def start_streaming(self):
#         if not self.peer_list.selectedItems():
#             self.log("ERROR: Please select a peer.")
#             return
        
#         model_type_str = self.model_type_combo.currentText()
#         target_ip = self.peer_list.selectedItems()[0].text().split('(')[-1].strip(')')
        
#         if not self.load_model() and "Uncompressed" not in model_type_str and "Codec" in model_type_str:
#             self.log("ERROR: Model could not be loaded. Aborting stream.")
#             return

#         self.log(f"--- Starting stream to {target_ip} with {model_type_str} ---")
        
#         self.streamer_thread = QThread()
#         self.streamer_worker = StreamerWorker(target_ip, self.model, model_type_str)
#         self.streamer_worker.moveToThread(self.streamer_thread)
#         self.streamer_worker.log_message.connect(self.log)
#         self.streamer_thread.started.connect(self.streamer_worker.run)
#         self.streamer_thread.start()
        
#         self.connect_button.setEnabled(False)
#         self.disconnect_button.setEnabled(True)
#         self.play_file_button.setEnabled(True)
#         self.model_type_combo.setEnabled(False)
#         self.status_label.setText("<b>Status:</b> <font color='green'>Connected</font>")

#     def stop_streaming(self):
#         if self.streamer_worker:
#             self.streamer_worker.stop()
#         if self.streamer_thread:
#             self.streamer_thread.quit()
#             self.streamer_thread.wait()
#         self.streamer_worker = None
#         self.streamer_thread = None
#         self.connect_button.setEnabled(True)
#         self.disconnect_button.setEnabled(False)
#         self.play_file_button.setEnabled(False)
#         self.model_type_combo.setEnabled(True)
#         self.log("Streaming stopped.")
#         self.status_label.setText("<b>Status:</b> <font color='red'>Disconnected</font>")

#     def closeEvent(self):
#         self.stop_streaming()
#         if hasattr(self, 'discovery_worker'):
#             self.discovery_worker.stop()
#         if hasattr(self, 'discovery_thread'):
#             self.discovery_thread.quit()
#             self.discovery_thread.wait()


import os
import threading
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit,
    QLabel, QFileDialog, QTextEdit, QSpinBox, QComboBox, 
    QGroupBox, QCheckBox, QListWidget
)
from PyQt5.QtCore import pyqtSignal, QObject, QThread, Qt
from PyQt5.QtGui import QFont

import socket
import pyaudio
import numpy as np
import torch
import time
import librosa
import struct

from model import (
    MuLawCodec, ALawCodec, DACCodec, EnCodecModel, TinyTransformerCodec, 
    HOP_SIZE, DAC_AVAILABLE, ENCODEC_AVAILABLE
)

# --- Configuration ---
BROADCAST_PORT = 37020
STREAM_PORT = 37021
DEVICE_ID = "NeuralCodecPC"
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = HOP_SIZE # 10ms (160 samples)

# --- Network Discovery Worker (Unchanged) ---
class DiscoveryWorker(QObject):
    peer_found = pyqtSignal(str, str)
    def __init__(self):
        super().__init__()
        self._running = True
        
    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(('', BROADCAST_PORT))
            except OSError as e:
                print(f"Discovery: Could not bind to port {BROADCAST_PORT}. {e}")
                return
            s.settimeout(1.0)
            while self._running:
                try:
                    data, addr = s.recvfrom(1024)
                    message = data.decode()
                    if message.startswith(DEVICE_ID):
                        self.peer_found.emit(message.split(':')[1], addr[0])
                except socket.timeout:
                    continue
                except Exception as e:
                    print(f"Discovery error: {e}")
                    
    def stop(self):
        self._running = False

# --- Audio Streaming Worker ---
class StreamerWorker(QObject):
    log_message = pyqtSignal(str)
    
    def __init__(self, target_ip, model, model_type_str):
        super().__init__()
        self.target_ip = target_ip
        self.model = model
        self.model_type_str = model_type_str
        self.p = pyaudio.PyAudio()
        self._running = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_muted = False
        self.file_playback_path = None
        self.file_playback_event = threading.Event()
        
        # Codec-specific buffering
        self.send_buffer = []
        self.recv_buffer = []
        
        if isinstance(self.model, DACCodec):
            self.chunk_size = 320 # DAC uses 20ms chunks
        elif isinstance(self.model, EnCodecModel):
            self.chunk_size = 480 # EnCodec may need different chunk size
        elif isinstance(self.model, TinyTransformerCodec):
            # Use the model's required window size (640 in trainer, but using low-latency 160 here is safer)
            # The model is trained on 640-sample chunks (40ms), so we must buffer up to 640 samples
            self.chunk_size = int(0.04 * RATE) # 640 samples (40ms chunk)
        else:
            self.chunk_size = CHUNK # Default 10ms

    def run(self):
        self.sender_thread = threading.Thread(target=self.send_audio, daemon=True)
        self.receiver_thread = threading.Thread(target=self.receive_audio, daemon=True)
        self.sender_thread.start()
        self.receiver_thread.start()

    def set_mute(self, muted):
        self.is_muted = muted
        self.log_message.emit(f"Microphone {'muted' if muted else 'unmuted'}.")

    def start_file_playback(self, filepath):
        self.file_playback_path = filepath
        self.file_playback_event.set()
        self.log_message.emit(f"Queued file for playback: {filepath}")

    def stop_file_playback(self, from_thread=False):
        self.file_playback_path = None
        if not from_thread:
            self.file_playback_event.clear()

    def encode_data(self, data_bytes):
        """Encodes raw audio bytes based on the selected model."""
        try:
            audio_np = np.frombuffer(data_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            if not self.model:
                return data_bytes
            
            if isinstance(self.model, (MuLawCodec, ALawCodec)):
                audio_tensor = torch.from_numpy(audio_np).unsqueeze(0).unsqueeze(0)
                encoded_tensor = self.model.encode(audio_tensor)
                return encoded_tensor.cpu().numpy().tobytes()
            
            # Grouping all neural codecs for chunked processing
            if isinstance(self.model, (DACCodec, EnCodecModel, TinyTransformerCodec)):
                
                # Buffer the raw audio
                self.send_buffer.extend(audio_np)
                
                # Check if we have enough data to meet the model's chunk requirement
                chunk_to_encode_size = self.chunk_size 
                
                if len(self.send_buffer) >= chunk_to_encode_size:
                    # Extract chunk and update buffer
                    chunk = np.array(self.send_buffer[:chunk_to_encode_size])
                    self.send_buffer = self.send_buffer[chunk_to_encode_size:]
                    
                    chunk_tensor = torch.from_numpy(chunk).unsqueeze(0).unsqueeze(0).to(self.device, dtype=torch.float32)
                    
                    with torch.no_grad():
                        
                        if isinstance(self.model, TinyTransformerCodec):
                            # VQ-Codec: returns quantized float codes AND the list of integer indices
                            _, indices_list, orig_len, _ = self.model.encode(chunk_tensor)
                            
                            # 1. Prepare payload: Concatenate integer indices (compressed data)
                            codes_np = torch.cat(indices_list, dim=1).cpu().numpy().astype(np.int32)
                            # 2. Set dtype to 0 (int32)
                            dtype_int = 0
                            
                        elif isinstance(self.model, DACCodec):
                            # DAC: Returns integer index tensor
                            codes_tensor, orig_len = self.model.encode(chunk_tensor)
                            codes_np = codes_tensor.cpu().numpy().astype(np.int32)
                            dtype_int = 0
                            
                        elif isinstance(self.model, EnCodecModel):
                            # EnCodec: Returns a tuple containing the Codes tensor which are integers
                            encoded_frames, orig_len = self.model.encode(chunk_tensor)
                            
                            # EnCodec format: (B, C, T) codes tensor
                            codes_tensor = encoded_frames[0] 
                            codes_np = codes_tensor.cpu().numpy().astype(np.int32)
                            dtype_int = 0
                        else:
                            return None

                        # Pack with metadata (orig_len, num_dims, dtype_int, shape...)
                        shape = codes_np.shape
                        header = struct.pack('I', orig_len) + struct.pack('I', len(shape)) + struct.pack('I', dtype_int)
                        for s in shape:
                            header += struct.pack('I', s)
                        
                        payload = header + codes_np.tobytes()
                        return payload
                else:
                    return None # Not enough data to encode
            
            return None # Should not be reached
        
        except Exception as e:
            self.log_message.emit(f"Encoding error: {e}")
            import traceback
            self.log_message.emit(traceback.format_exc())
            return None

    def play_file_audio(self, filepath, s):
        try:
            wav, _ = librosa.load(filepath, sr=RATE, mono=True)
            self.log_message.emit(f"Sending file: {filepath}...")
            wav_int16 = (wav * 32767.0).astype(np.int16)
            
            self.send_buffer = []
            
            # Chunking by the CHUNK (10ms=160 samples) for micro-buffering, but encoding by self.chunk_size (40ms=640 samples)
            for i in range(0, len(wav_int16), CHUNK):
                if not self._running or self.file_playback_path != filepath:
                    break
                
                chunk_data = wav_int16[i:i+CHUNK]
                if len(chunk_data) < CHUNK:
                    chunk_data = np.pad(chunk_data, (0, CHUNK - len(chunk_data)), 'constant')
                
                # Convert 10ms int16 chunk to bytes and encode (which will buffer it until 640 samples are met)
                payload = self.encode_data(chunk_data.tobytes())
                
                if payload:
                    s.sendto(payload, (self.target_ip, STREAM_PORT))
                
                time.sleep(float(CHUNK) / RATE) # Stream rate control
                
            self.log_message.emit("File playback finished.")
            self.send_buffer = []
            
        except Exception as e:
            self.log_message.emit(f"Error playing file: {e}")
            self.stop_file_playback(from_thread=True)

    def send_audio(self):
        try:
            stream = self.p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                while self._running:
                    if self.file_playback_event.is_set():
                        self.file_playback_event.clear()
                        if self.file_playback_path:
                            self.play_file_audio(self.file_playback_path, s)
                        if self.file_playback_path:
                            self.file_playback_path = None
                        continue
                        
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    payload = self.encode_data(b'\x00' * (CHUNK * 2) if self.is_muted else data)
                    
                    if payload:
                        s.sendto(payload, (self.target_ip, STREAM_PORT))
                        
        except Exception as e:
            if self._running:
                self.log_message.emit(f"ERROR in send_audio: {e}")
        finally:
            if hasattr(self, 'p'):
                if 'stream' in locals() and stream.is_active():
                    stream.stop_stream()
                    stream.close()
                self.p.terminate()

    def receive_audio(self):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.bind(('', STREAM_PORT))
                s.settimeout(2.0)
                stream = self.p.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True, frames_per_buffer=CHUNK)
                
                self.recv_buffer = []
                
                while self._running:
                    try:
                        data, _ = s.recvfrom(65536)
                    except socket.timeout:
                        if self._running:
                            self.log_message.emit("Socket timeout, resetting decoder state.")
                        self.recv_buffer = []
                        continue
                        
                    if not self.model:
                        payload = data
                    else:
                        try:
                            if isinstance(self.model, (MuLawCodec, ALawCodec)):
                                latent_tensor = torch.from_numpy(np.frombuffer(data, dtype=np.uint8)).unsqueeze(0)
                                decoded_tensor = self.model.decode(latent_tensor)
                                payload = (decoded_tensor.squeeze().cpu().numpy() * 32767.0).astype(np.int16).tobytes()
                                
                            elif isinstance(self.model, (DACCodec, EnCodecModel, TinyTransformerCodec)): 
                                # Neural codec decoding (handling VQ indices)
                                try:
                                    # Unpack header: orig_len (I), num_dims (I), dtype (I)
                                    offset = 0
                                    orig_len = struct.unpack('I', data[offset:offset+4])[0]
                                    offset += 4
                                    
                                    num_dims = struct.unpack('I', data[offset:offset+4])[0]
                                    offset += 4
                                    
                                    dtype_int = struct.unpack('I', data[offset:offset+4])[0]
                                    offset += 4
                                    
                                    shape = []
                                    for _ in range(num_dims):
                                        shape.append(struct.unpack('I', data[offset:offset+4])[0])
                                        offset += 4
                                    
                                    codes_bytes = data[offset:]
                                    
                                    # Determine numpy dtype based on header (1=float32, 0=int32)
                                    dtype = np.float32 if dtype_int == 1 else np.int32 # Should be int32 for VQ
                                    
                                    # Reconstruct codes tensor
                                    codes_np = np.frombuffer(codes_bytes, dtype=dtype)
                                    if shape:
                                        codes_np = codes_np.reshape(shape)
                                    
                                    codes_tensor = torch.from_numpy(codes_np).to(self.device)
                                    
                                    with torch.no_grad():
                                        if isinstance(self.model, TinyTransformerCodec):
                                            # VQ-Codec: Convert concatenated indices back to list of tensors (B*T, C) -> list of (B*T, 1)
                                            # codes_tensor shape is (B, C, T) or (N, C)
                                            
                                            # We are transmitting the indices list concatenated into one tensor: (N_frames * B, Num_Codebooks)
                                            N_frames = codes_tensor.shape[0] * codes_tensor.shape[1] # Should be B*T_latent
                                            Num_Codebooks = self.model.num_codebooks
                                            
                                            # Reshape to (N_frames, Num_Codebooks) if necessary, then split
                                            
                                            # If shape is (B, Num_Codebooks, T), reshape to (B*T, Num_Codebooks)
                                            if codes_tensor.dim() == 3:
                                                codes_tensor = codes_tensor.permute(0, 2, 1).contiguous().view(-1, Num_Codebooks)
                                            
                                            indices_list = codes_tensor.chunk(Num_Codebooks, dim=1)
                                            decoded_tensor = self.model.decode(indices_list, orig_len, encoder_outputs=None)
                                        
                                        elif isinstance(self.model, DACCodec):
                                            # DAC uses its internal logic for decoding integer codes
                                            decoded_tensor = self.model.decode(codes_tensor, orig_len)
                                        
                                        elif isinstance(self.model, EnCodecModel):
                                            # EnCodec expects a tuple containing the codes tensor
                                            decoded_tensor = self.model.decode([codes_tensor], orig_len)

                                    decoded_audio = decoded_tensor.squeeze().cpu().numpy()
                                    self.recv_buffer.extend(decoded_audio)
                                    
                                    # Play back audio when enough data is buffered (using the default CHUNK)
                                    while len(self.recv_buffer) >= CHUNK:
                                        output_chunk = np.array(self.recv_buffer[:CHUNK])
                                        self.recv_buffer = self.recv_buffer[CHUNK:]
                                        payload = (output_chunk * 32767.0).astype(np.int16).tobytes()
                                        stream.write(payload)
                                        continue
                                        
                                except Exception as codec_err:
                                    self.log_message.emit(f"Codec decode error: {codec_err}")
                                    continue
                        except Exception as e:
                            self.log_message.emit(f"Decoding error: {e}")
                            continue
                        
                    stream.write(payload)
        except Exception as e:
            if self._running:
                self.log_message.emit(f"ERROR in receive_audio: {e}")
        finally:
            if hasattr(self, 'p'):
                if 'stream' in locals() and stream.is_active():
                    stream.stop_stream()
                    stream.close()
                self.p.terminate()

    def stop(self):
        self._running = False
        self.stop_file_playback()


class StreamingTab(QWidget):
    def __init__(self):
        super().__init__()
        self.peers = {}
        self.streamer_worker = None
        self.streamer_thread = None
        self._setup_ui()
        self._start_discovery()
        self.model = None

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        config_group = QGroupBox("Configuration")
        config_layout = QVBoxLayout(config_group)

        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("<b>Codec:</b>"))
        self.model_type_combo = QComboBox()
        
        # All models listed here: Baselines, Custom, DAC, EnCodec
        model_items = [
            "Uncompressed",
            "μ-Law Codec",  
            "A-Law Codec",
            "Tiny Transformer Codec (Custom, 16kHz, ~10ms)", 
        ]
        if DAC_AVAILABLE:
            model_items.append("DAC Codec (16kHz, 20ms)")
        if ENCODEC_AVAILABLE:
            model_items.append("EnCodec (24kHz, 6kbps)")
        
        self.model_type_combo.addItems(model_items)
        self.model_type_combo.currentTextChanged.connect(self.on_model_type_changed)
        model_layout.addWidget(self.model_type_combo)
        config_layout.addLayout(model_layout)
        
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("Path to TRAINED model (.pth). Mandatory for Tiny Transformer Codec.") 
        self.browse_model_button = QPushButton("Browse...")
        self.browse_model_button.clicked.connect(self.browse_model)
        self.load_model_button = QPushButton("Load Model")
        self.load_model_button.clicked.connect(self.load_model)
        model_path_layout = QHBoxLayout()
        model_path_layout.addWidget(self.model_path_edit)
        model_path_layout.addWidget(self.browse_model_button)
        model_path_layout.addWidget(self.load_model_button)
        config_layout.addLayout(model_path_layout)
        
        self.bitrate_label = QLabel(f"Codec Info: Select a codec")
        config_layout.addWidget(self.bitrate_label)

        self.play_file_path_edit = QLineEdit()
        self.play_file_path_edit.setPlaceholderText("Path to audio file for playback...")
        self.browse_play_file_button = QPushButton("Browse...")
        self.browse_play_file_button.clicked.connect(self.browse_play_file)
        file_playback_layout = QHBoxLayout()
        file_playback_layout.addWidget(self.play_file_path_edit)
        file_playback_layout.addWidget(self.browse_play_file_button)
        config_layout.addLayout(file_playback_layout)
        layout.addWidget(config_group)

        network_group = QGroupBox("Network")
        network_layout = QVBoxLayout(network_group)
        network_layout.addWidget(QLabel("<b>Available Peers:</b>"))
        self.peer_list = QListWidget()
        self.peer_list.setMaximumHeight(100)
        self.refresh_button = QPushButton("Refresh List")
        self.refresh_button.clicked.connect(self.send_broadcast)
        network_layout.addWidget(self.peer_list)
        network_layout.addWidget(self.refresh_button, Qt.AlignRight)
        layout.addWidget(network_group)

        controls_group = QGroupBox("Controls")
        controls_layout = QHBoxLayout(controls_group)
        self.connect_button = QPushButton("Start Streaming")
        self.connect_button.setStyleSheet("background-color: #4CAF50; color: white; padding: 5px; border-radius: 5px;")
        self.connect_button.clicked.connect(self.start_streaming)
        self.disconnect_button = QPushButton("Stop Streaming")
        self.disconnect_button.setStyleSheet("background-color: #f44336; color: white; padding: 5px; border-radius: 5px;")
        self.disconnect_button.setEnabled(False)
        self.disconnect_button.clicked.connect(self.stop_streaming)
        self.play_file_button = QPushButton("▶ Play File")
        self.play_file_button.setEnabled(False)
        self.play_file_button.clicked.connect(self.start_file_playback)
        self.mute_mic_checkbox = QCheckBox("Mute Mic")
        self.mute_mic_checkbox.stateChanged.connect(self.on_mute_changed)
        self.status_label = QLabel("<b>Status:</b> <font color='red'>Disconnected</font>")
        
        controls_layout.addWidget(self.connect_button)
        controls_layout.addWidget(self.disconnect_button)
        controls_layout.addWidget(self.play_file_button)
        controls_layout.addStretch()
        controls_layout.addWidget(self.mute_mic_checkbox)
        controls_layout.addWidget(self.status_label)
        layout.addWidget(controls_group)

        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)
        log_layout.addWidget(self.log_text_edit)
        layout.addWidget(log_group)
        
        self.on_model_type_changed(self.model_type_combo.currentText())

    def log(self, message):
        self.log_text_edit.append(message)
        self.log_text_edit.verticalScrollBar().setValue(self.log_text_edit.verticalScrollBar().maximum())

    def on_model_type_changed(self, model_name):
        is_neural = any(x in model_name for x in ["DAC", "EnCodec", "Transformer"])
        is_custom_transformer = "Tiny Transformer Codec" in model_name
        
        self.model_path_edit.setEnabled(is_neural)
        self.browse_model_button.setEnabled(is_neural)
        self.load_model_button.setEnabled(is_neural)

        if not is_neural:
            self.model_path_edit.setText("N/A (Traditional or Uncompressed)")
            bitrate = '256 kbps' if 'Uncompressed' in model_name else '128 kbps'
            self.bitrate_label.setText(f"Codec Bitrate: {bitrate}")
            self.load_model()
        elif "DAC" in model_name:
            self.model_path_edit.setText("(auto-download if not provided)")
            self.bitrate_label.setText(f"Codec: DAC | Bitrate: ~8-12 kbps | Latency: 20ms")
        elif "EnCodec" in model_name:
            self.model_path_edit.setText("(auto-download if not provided)")
            self.bitrate_label.setText(f"Codec: EnCodec | Bitrate: 6 kbps | Latency: ~10ms")
        elif is_custom_transformer:
            self.model_path_edit.setText("MANDATORY: Path to your trained checkpoint file (.pth)")
            self.bitrate_label.setText(f"Codec: Tiny Transformer | Bitrate: 9.00 kbps (Calculated) | Latency: 20ms") # Updated bitrate

    def on_mute_changed(self, state):
        if self.streamer_worker:
            self.streamer_worker.set_mute(state == Qt.Checked)

    def browse_play_file(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Select Audio File", "", "Audio Files (*.wav *.mp3)")
        if filepath:
            self.play_file_path_edit.setText(filepath)

    def start_file_playback(self):
        if not self.play_file_path_edit.text():
            self.log("ERROR: Please select a file to play.")
            return
        if self.streamer_worker:
            self.streamer_worker.stop_file_playback()
            self.streamer_worker.start_file_playback(self.play_file_path_edit.text())
        else:
            self.log("ERROR: Must be streaming to play a file.")

    def browse_model(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Select Model File", "", "PyTorch Models (*.pth *.pt)")
        if filepath:
            self.model_path_edit.setText(filepath)

    def load_model(self):
        model_type_str = self.model_type_combo.currentText()
        self.model = None

        if "Uncompressed" in model_type_str:
            self.model = None
            self.log("Streaming uncompressed (256 kbps).")
            return True
        elif "μ-Law" in model_type_str:
            self.model = MuLawCodec()
            self.log("Loaded μ-Law Codec (128 kbps).")
            return True
        elif "A-Law" in model_type_str:
            self.model = ALawCodec()
            self.log("Loaded A-Law Codec (128 kbps).")
            return True
        elif "DAC" in model_type_str:
            if not DAC_AVAILABLE:
                self.log("ERROR: DAC not installed. Install with: pip install descript-audio-codec")
                return False
            try:
                path = self.model_path_edit.text()
                if "auto-download" in path or not path or "N/A" in path:
                    path = None
                self.model = DACCodec(model_path=path, model_type="16khz")
                self.log("Loaded DAC Codec.")
                return True
            except Exception as e:
                self.log(f"ERROR: Failed to load DAC: {e}")
                return False
        elif "EnCodec" in model_type_str:
            if not ENCODEC_AVAILABLE:
                self.log("ERROR: EnCodec not installed. Install with: pip install encodec")
                return False
            try:
                path = self.model_path_edit.text()
                if "auto-download" in path or not path or "N/A" in path:
                    path = None
                self.model = EnCodecModel(model_path=path)
                self.log("Loaded EnCodec.")
                return True
            except Exception as e:
                self.log(f"ERROR: Failed to load EnCodec: {e}")
                return False
        elif "Tiny Transformer Codec" in model_type_str: # Custom Codec Loading
            model_path = self.model_path_edit.text()
            if "MANDATORY" in model_path or not os.path.exists(model_path):
                self.log("ERROR: Please provide a valid path to your trained Tiny Transformer Codec (.pth) model.")
                return False
            try:
                self.model = TinyTransformerCodec.load_model(model_path)
                self.log("Loaded Tiny Transformer Codec.")
                return True
            except Exception as e:
                self.log(f"ERROR: Failed to load Tiny Transformer Codec: {e}")
                return False
            
        return False

    def add_peer(self, name, ip):
        if ip not in self.peers.values():
            self.peers[name] = ip
            self.peer_list.addItem(f"{name} ({ip})")

    def send_broadcast(self):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                s.sendto(f"{DEVICE_ID}:{socket.gethostname()}".encode(), ('<broadcast>', BROADCAST_PORT))
            self.log("Sent discovery broadcast.")
        except Exception as e:
            self.log(f"Error sending broadcast: {e}")

    def _start_discovery(self):
        self.discovery_thread = QThread()
        self.discovery_worker = DiscoveryWorker()
        self.discovery_worker.moveToThread(self.discovery_thread)
        self.discovery_worker.peer_found.connect(self.add_peer)
        self.discovery_thread.started.connect(self.discovery_worker.run)
        self.discovery_thread.start()
        self.send_broadcast()

    def start_streaming(self):
        if not self.peer_list.selectedItems():
            self.log("ERROR: Please select a peer.")
            return
        
        model_type_str = self.model_type_combo.currentText()
        target_ip = self.peer_list.selectedItems()[0].text().split('(')[-1].strip(')')
        
        if not self.load_model() and "Uncompressed" not in model_type_str and "Codec" in model_type_str:
            self.log("ERROR: Model could not be loaded. Aborting stream.")
            return

        self.log(f"--- Starting stream to {target_ip} with {model_type_str} ---")
        
        self.streamer_thread = QThread()
        self.streamer_worker = StreamerWorker(target_ip, self.model, model_type_str)
        self.streamer_worker.moveToThread(self.streamer_thread)
        self.streamer_worker.log_message.connect(self.log)
        self.streamer_thread.started.connect(self.streamer_worker.run)
        self.streamer_thread.start()
        
        self.connect_button.setEnabled(False)
        self.disconnect_button.setEnabled(True)
        self.play_file_button.setEnabled(True)
        self.model_type_combo.setEnabled(False)
        self.status_label.setText("<b>Status:</b> <font color='green'>Connected</font>")

    def stop_streaming(self):
        if self.streamer_worker:
            self.streamer_worker.stop()
        if self.streamer_thread:
            self.streamer_thread.quit()
            self.streamer_thread.wait()
        self.streamer_worker = None
        self.streamer_thread = None
        self.connect_button.setEnabled(True)
        self.disconnect_button.setEnabled(False)
        self.play_file_button.setEnabled(False)
        self.model_type_combo.setEnabled(True)
        self.log("Streaming stopped.")
        self.status_label.setText("<b>Status:</b> <font color='red'>Disconnected</font>")

    def closeEvent(self):
        self.stop_streaming()
        if hasattr(self, 'discovery_worker'):
            self.discovery_worker.stop()
        if hasattr(self, 'discovery_thread'):
            self.discovery_thread.quit()
            self.discovery_thread.wait()
