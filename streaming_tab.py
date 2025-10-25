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
    MuLawCodec, ALawCodec, TS3_Codec, DACCodec,
    HOP_SIZE, NUM_VQ_STAGES, VQ_INDICES_PER_STAGE, DAC_AVAILABLE
)

# --- Configuration ---
BROADCAST_PORT = 37020
STREAM_PORT = 37021
DEVICE_ID = "NeuralCodecPC"
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = HOP_SIZE # 10ms (160 samples) for microphone input
COMPRESSED_PAYLOAD_SIZE = NUM_VQ_STAGES * VQ_INDICES_PER_STAGE

# --- Network Discovery Worker ---
class DiscoveryWorker(QObject):
    peer_found = pyqtSignal(str, str)
    def __init__(self): super().__init__(); self._running = True
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
                except socket.timeout: continue
                except Exception as e: print(f"Discovery error: {e}")
    def stop(self): self._running = False

# --- Audio Streaming Worker ---
class StreamerWorker(QObject):
    log_message = pyqtSignal(str)
    def __init__(self, target_ip, model, model_type_str, tfm_history_chunks):
        super().__init__()
        self.target_ip, self.model, self.model_type_str = target_ip, model, model_type_str
        self.p = pyaudio.PyAudio()
        self._running = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_muted = False
        self.file_playback_path = None
        self.file_playback_event = threading.Event()
        self.tfm_history_chunks = tfm_history_chunks
        
        self.h_enc, self.h_dec = None, None
        
        # DAC-specific buffering is now 320 samples (20ms)
        self.dac_send_buffer = []
        self.dac_recv_buffer = []
        # DAC codec uses 320 sample chunks, TS3 uses 160. CHUNK is the microphone read size (160).
        self.dac_chunk_size = 320 
        
        # Re-instantiate TS3 model with history on the device
        if self.model and isinstance(self.model, TS3_Codec):
            # If a model was loaded, it needs to be moved to the correct device
            # and re-configured for streaming history chunks if applicable.
            self.model.to(self.device)
            self.model.eval()
        elif self.model and isinstance(self.model, DACCodec):
            self.log_message.emit(f"DAC streaming will use {self.dac_chunk_size} sample chunks ({self.dac_chunk_size/16000*1000:.0f}ms latency)")

    def run(self):
        self.sender_thread = threading.Thread(target=self.send_audio, daemon=True)
        self.receiver_thread = threading.Thread(target=self.receive_audio, daemon=True)
        self.sender_thread.start(); self.receiver_thread.start()
        # Sender/Receiver run indefinitely until stopped, so we don't need to join them here.

    def set_mute(self, muted):
        self.is_muted = muted
        self.log_message.emit(f"Microphone {'muted' if muted else 'unmuted'}.")

    def start_file_playback(self, filepath):
        self.file_playback_path = filepath
        self.file_playback_event.set()
        self.log_message.emit(f"Queued file for playback: {filepath}")

    def stop_file_playback(self): 
        # Setting file_playback_path to None and clearing event ensures the send_audio loop stops file mode.
        self.file_playback_path = None
        self.file_playback_event.clear()

    def encode_data(self, data_bytes):
        """Encodes raw audio bytes based on the selected model."""
        try:
            # Mic reads 160 samples (10ms) chunk
            audio_np = np.frombuffer(data_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            if not self.model:
                return data_bytes
            
            if isinstance(self.model, (MuLawCodec, ALawCodec)):
                audio_tensor = torch.from_numpy(audio_np).unsqueeze(0).unsqueeze(0)
                encoded_tensor = self.model.encode(audio_tensor)
                return encoded_tensor.cpu().numpy().tobytes()
            
            if isinstance(self.model, DACCodec):
                # DAC encoding - Buffer CHUNK (160) until we have DAC_CHUNK_SIZE (320) samples
                self.dac_send_buffer.extend(audio_np)
                
                if len(self.dac_send_buffer) >= self.dac_chunk_size:
                    # Process accumulated samples
                    chunk = np.array(self.dac_send_buffer[:self.dac_chunk_size])
                    self.dac_send_buffer = self.dac_send_buffer[self.dac_chunk_size:]
                    
                    chunk_tensor = torch.from_numpy(chunk).unsqueeze(0).unsqueeze(0).to(self.device, dtype=torch.float32)
                    with torch.no_grad():
                        codes, orig_len = self.model.encode(chunk_tensor)
                        
                        # Serialize codes for network transmission
                        # codes shape: (1, n_codebooks, seq_len)
                        codes_np = codes.cpu().numpy().astype(np.int32)
                        
                        # Pack: [orig_len (4 bytes)] + [shape info] + [codes data]
                        # We include the original length for trimming on the decode side
                        shape = codes_np.shape
                        header = struct.pack('IIII', orig_len, shape[0], shape[1], shape[2])
                        payload = header + codes_np.tobytes()
                        return payload
                else:
                    # Not enough data yet, return None
                    return None

            # TS3 RVQ encoding (10ms chunk)
            audio_tensor = torch.from_numpy(audio_np).unsqueeze(0).unsqueeze(0).to(self.device, dtype=torch.float32)
            with torch.no_grad():
                indices, self.h_enc = self.model.encode(audio_tensor, self.h_enc)
                payload_bytes = indices.cpu().numpy().astype(np.uint8).tobytes()
                return payload_bytes

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
            
            # Reset states for file playback
            self.h_enc = None
            self.dac_send_buffer = [] 
            
            # Use CHUNK (160 samples/10ms) as the sending interval even for DAC
            for i in range(0, len(wav_int16), CHUNK):
                if not self._running or self.file_playback_path != filepath: break
                
                chunk_data = wav_int16[i:i+CHUNK]
                if len(chunk_data) < CHUNK:
                    chunk_data = np.pad(chunk_data, (0, CHUNK - len(chunk_data)), 'constant')
                
                chunk_bytes = chunk_data.tobytes()
                
                # encode_data will handle buffering/chunking if DAC is selected
                payload = self.encode_data(chunk_bytes) 
                
                if payload: 
                    s.sendto(payload, (self.target_ip, STREAM_PORT))
                
                # Maintain real-time rhythm (10ms per loop iteration)
                time.sleep(float(CHUNK) / RATE)
                
            self.log_message.emit("File playback finished.")
            self.h_enc = None
            self.dac_send_buffer = []
            
        except Exception as e: 
            self.log_message.emit(f"Error playing file: {e}")
            self.stop_file_playback()

    def send_audio(self):
        try:
            stream = self.p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                while self._running:
                    # Handle file playback precedence
                    if self.file_playback_event.is_set():
                        self.file_playback_event.clear()
                        if self.file_playback_path: self.play_file_audio(self.file_playback_path, s)
                        if self.file_playback_path: self.file_playback_path = None # Clear path after playback
                        continue
                    
                    # Live Mic Streaming
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    payload = self.encode_data(b'\x00' * (CHUNK * 2) if self.is_muted else data)
                    
                    if payload: 
                        s.sendto(payload, (self.target_ip, STREAM_PORT))
                        
        except Exception as e: 
            if self._running: self.log_message.emit(f"ERROR in send_audio: {e}")
        finally:
            if hasattr(self, 'p'): 
                if 'stream' in locals() and stream.is_active(): stream.stop_stream(); stream.close()
                self.p.terminate()

    def receive_audio(self):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.bind(('', STREAM_PORT))
                s.settimeout(2.0)
                stream = self.p.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True, frames_per_buffer=CHUNK)
                
                self.h_dec = None
                self.dac_recv_buffer = []
                
                while self._running:
                    try:
                        data, _ = s.recvfrom(65536)
                    except socket.timeout:
                        if self._running: self.log_message.emit("Socket timeout, resetting decoder state.")
                        self.h_dec = None
                        self.dac_recv_buffer = []
                        continue
                        
                    if not self.model:
                        payload = data
                    else:
                        try:
                            if isinstance(self.model, (MuLawCodec, ALawCodec)):
                                latent_tensor = torch.from_numpy(np.frombuffer(data, dtype=np.uint8)).unsqueeze(0)
                                decoded_tensor = self.model.decode(latent_tensor)
                            elif isinstance(self.model, DACCodec):
                                # DAC decoding
                                try:
                                    # Unpack header: orig_len, shape[0], shape[1], shape[2]
                                    if len(data) < 16:
                                        self.log_message.emit(f"DAC packet too small: {len(data)} bytes")
                                        continue
                                        
                                    orig_len, shape0, shape1, shape2 = struct.unpack('IIII', data[:16])
                                    codes_bytes = data[16:]
                                    
                                    # Check size against expected total size
                                    expected_size = shape0 * shape1 * shape2 * 4 # int32 = 4 bytes
                                    if len(codes_bytes) != expected_size:
                                        self.log_message.emit(f"DAC packet size mismatch. Expected {expected_size}, got {len(codes_bytes)}")
                                        continue
                                        
                                    codes_np = np.frombuffer(codes_bytes, dtype=np.int32).reshape(shape0, shape1, shape2)
                                    codes = torch.from_numpy(codes_np).to(self.device)
                                    
                                    with torch.no_grad():
                                        decoded_tensor = self.model.decode(codes, orig_len)
                                    
                                    # Add to buffer
                                    decoded_audio = decoded_tensor.squeeze().cpu().numpy()
                                    self.dac_recv_buffer.extend(decoded_audio)
                                    
                                    # Output in CHUNK-sized pieces (160 samples, 10ms) for smoother playback rhythm
                                    while len(self.dac_recv_buffer) >= CHUNK:
                                        output_chunk = np.array(self.dac_recv_buffer[:CHUNK])
                                        self.dac_recv_buffer = self.dac_recv_buffer[CHUNK:]
                                        payload = (output_chunk * 32767.0).astype(np.int16).tobytes()
                                        stream.write(payload)
                                    continue # Skip the normal stream.write below
                                    
                                except Exception as dac_err:
                                    self.log_message.emit(f"DAC decode error: {dac_err}")
                                    import traceback
                                    self.log_message.emit(traceback.format_exc())
                                    continue
                            else:
                                # TS3 decoding
                                with torch.no_grad():
                                    if len(data) != COMPRESSED_PAYLOAD_SIZE:
                                        self.log_message.emit(f"Corrupted packet. Resetting state.")
                                        self.h_dec = None
                                        continue
                                        
                                    indices = torch.from_numpy(np.frombuffer(data, dtype=np.uint8)).unsqueeze(0).to(self.device, dtype=torch.long)
                                    decoded_tensor, self.h_dec = self.model.decode(indices, self.h_dec)
                                    
                                decoded_tensor = decoded_tensor[..., :CHUNK]
                                payload = (decoded_tensor.squeeze().cpu().numpy() * 32767.0).astype(np.int16).tobytes()

                        except Exception as e: 
                            self.log_message.emit(f"Decoding error: {e}")
                            import traceback
                            self.log_message.emit(traceback.format_exc())
                            continue
                        
                    stream.write(payload)
        except Exception as e:
            if self._running: self.log_message.emit(f"ERROR in receive_audio: {e}")
        finally:
            if hasattr(self, 'p'): 
                if 'stream' in locals() and stream.is_active(): stream.stop_stream(); stream.close()
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
        
        model_items = [
            "Uncompressed",
            "μ-Law Codec", 
            "A-Law Codec",
            "TS3_Codec (16kbps GACodec)"
        ]
        if DAC_AVAILABLE:
            model_items.append("DAC Codec (16kHz, 20ms Latency)")
        
        self.model_type_combo.addItems(model_items)
        self.model_type_combo.currentTextChanged.connect(self.on_model_type_changed)
        model_layout.addWidget(self.model_type_combo)
        config_layout.addLayout(model_layout)
        
        self.model_path_edit = QLineEdit(); self.model_path_edit.setPlaceholderText("Path to trained Generator (.pth)...")
        self.browse_model_button = QPushButton("Browse..."); self.browse_model_button.clicked.connect(self.browse_model)
        self.load_model_button = QPushButton("Load Model"); self.load_model_button.clicked.connect(self.load_model)
        model_path_layout = QHBoxLayout()
        model_path_layout.addWidget(self.model_path_edit); model_path_layout.addWidget(self.browse_model_button); model_path_layout.addWidget(self.load_model_button)
        config_layout.addLayout(model_path_layout)
        
        lat_h_layout = QHBoxLayout()
        self.tfm_history_spinbox = QSpinBox(); self.tfm_history_spinbox.setRange(0, 5); self.tfm_history_spinbox.setValue(0)
        self.bitrate_label = QLabel(f"Codec Bitrate: 16 kbps | Frame Latency: {HOP_SIZE/16000*1000:.0f} ms")
        lat_h_layout.addWidget(QLabel("TFM History Chunks:")); lat_h_layout.addWidget(self.tfm_history_spinbox)
        lat_h_layout.addWidget(self.bitrate_label)
        config_layout.addLayout(lat_h_layout)

        self.play_file_path_edit = QLineEdit(); self.play_file_path_edit.setPlaceholderText("Path to audio file for playback...")
        self.browse_play_file_button = QPushButton("Browse..."); self.browse_play_file_button.clicked.connect(self.browse_play_file)
        file_playback_layout = QHBoxLayout()
        file_playback_layout.addWidget(self.play_file_path_edit); file_playback_layout.addWidget(self.browse_play_file_button)
        config_layout.addLayout(file_playback_layout)
        layout.addWidget(config_group)

        network_group = QGroupBox("Network")
        network_layout = QVBoxLayout(network_group)
        network_layout.addWidget(QLabel("<b>Available Peers:</b>"))
        self.peer_list = QListWidget(); self.peer_list.setMaximumHeight(100)
        self.refresh_button = QPushButton("Refresh List"); self.refresh_button.clicked.connect(self.send_broadcast)
        network_layout.addWidget(self.peer_list); network_layout.addWidget(self.refresh_button, Qt.AlignRight)
        layout.addWidget(network_group)

        controls_group = QGroupBox("Controls")
        controls_layout = QHBoxLayout(controls_group)
        self.connect_button = QPushButton("Start Streaming"); self.connect_button.setStyleSheet("background-color: #4CAF50; color: white; padding: 5px; border-radius: 5px;"); self.connect_button.clicked.connect(self.start_streaming)
        self.disconnect_button = QPushButton("Stop Streaming"); self.disconnect_button.setStyleSheet("background-color: #f44336; color: white; padding: 5px; border-radius: 5px;"); self.disconnect_button.setEnabled(False); self.disconnect_button.clicked.connect(self.stop_streaming)
        self.play_file_button = QPushButton("▶ Play File"); self.play_file_button.setEnabled(False); self.play_file_button.clicked.connect(self.start_file_playback)
        self.mute_mic_checkbox = QCheckBox("Mute Mic"); self.mute_mic_checkbox.stateChanged.connect(self.on_mute_changed)
        self.status_label = QLabel("<b>Status:</b> <font color='red'>Disconnected</font>")
        
        controls_layout.addWidget(self.connect_button); controls_layout.addWidget(self.disconnect_button); controls_layout.addWidget(self.play_file_button)
        controls_layout.addStretch(); controls_layout.addWidget(self.mute_mic_checkbox); controls_layout.addWidget(self.status_label)
        layout.addWidget(controls_group)

        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        self.log_text_edit = QTextEdit(); self.log_text_edit.setReadOnly(True)
        log_layout.addWidget(self.log_text_edit)
        layout.addWidget(log_group)
        
        self.on_model_type_changed(self.model_type_combo.currentText())

    def log(self, message): 
        self.log_text_edit.append(message)
        self.log_text_edit.verticalScrollBar().setValue(self.log_text_edit.verticalScrollBar().maximum())

    def on_model_type_changed(self, model_name):
        is_neural = "TS3_Codec" in model_name
        is_dac = "DAC" in model_name
        
        self.model_path_edit.setEnabled(is_neural or is_dac)
        self.browse_model_button.setEnabled(is_neural or is_dac)
        self.load_model_button.setEnabled(is_neural or is_dac)
        self.tfm_history_spinbox.setEnabled(is_neural)

        if not is_neural and not is_dac:
            self.model_path_edit.setText("N/A (Traditional or Uncompressed)")
            bitrate = '256 kbps (Uncomp)' if 'Uncompressed' in model_name else '128 kbps (Law)'
            self.bitrate_label.setText(f"Codec Bitrate: {bitrate} | Frame Latency: {HOP_SIZE/16000*1000:.0f} ms")
            self.load_model()
        elif is_dac:
            self.model_path_edit.setText("weights_16khz.pth (auto-download)")
            self.bitrate_label.setText(f"Codec Bitrate: ~8-12 kbps (DAC) | Frame Latency: 20 ms")
        else:
            self.model_path_edit.setText("low_latency_codec_ts3_gacodec.pth")
            self.bitrate_label.setText(f"Codec Bitrate: 16 kbps | Frame Latency: {HOP_SIZE/16000*1000:.0f} ms")

    def on_mute_changed(self, state):
        if self.streamer_worker: self.streamer_worker.set_mute(state == Qt.Checked)

    def browse_play_file(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Select Audio File", "", "Audio Files (*.wav *.mp3)")
        if filepath: self.play_file_path_edit.setText(filepath)

    def start_file_playback(self):
        if not self.play_file_path_edit.text(): self.log("ERROR: Please select a file to play."); return
        if self.streamer_worker: 
            # Stops previous playback if active
            self.streamer_worker.stop_file_playback() 
            self.streamer_worker.start_file_playback(self.play_file_path_edit.text())
        else: self.log("ERROR: Must be streaming to play a file.")

    def browse_model(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Select Model File", "", "PyTorch Models (*.pth)")
        if filepath: self.model_path_edit.setText(filepath)

    def load_model(self):
        model_type_str = self.model_type_combo.currentText()
        self.model = None

        if "Uncompressed" in model_type_str: 
            self.model = None; self.log("Streaming uncompressed (256 kbps)."); return True
        if "μ-Law" in model_type_str: 
            self.model = MuLawCodec(); self.log("Loaded μ-Law Codec (128 kbps)."); return True
        if "A-Law" in model_type_str: 
            self.model = ALawCodec(); self.log("Loaded A-Law Codec (128 kbps)."); return True
        if "DAC" in model_type_str:
            if not DAC_AVAILABLE:
                self.log("ERROR: DAC not installed. Install with: pip install descript-audio-codec")
                return False
            try:
                path = self.model_path_edit.text()
                if "auto-download" in path or not path or "N/A" in path:
                    path = None
                self.model = DACCodec(model_path=path, model_type="16khz")
                self.log("Loaded DAC Codec (16kHz, 20ms).")
                return True
            except Exception as e:
                self.log(f"ERROR: Failed to load DAC: {e}")
                return False
        
        try:
            path = self.model_path_edit.text()
            if not path or "N/A" in path: 
                self.model = None; return False
            
            if "TS3_Codec" in model_type_str: 
                self.model = TS3_Codec(history_chunks=self.tfm_history_spinbox.value())
                self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
                self.model.eval()
            else: 
                self.model = None; return False

            self.log(f"Successfully checked model file: {path}"); return True
        except Exception as e:
            self.log(f"ERROR: Failed to load model. {e}"); self.model = None; return False

    def add_peer(self, name, ip):
        if ip not in self.peers.values(): self.peers[name] = ip; self.peer_list.addItem(f"{name} ({ip})")

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
        self.discovery_thread.start(); self.send_broadcast()

    def start_streaming(self):
        if not self.peer_list.selectedItems(): self.log("ERROR: Please select a peer."); return
        
        model_type_str = self.model_type_combo.currentText()
        target_ip = self.peer_list.selectedItems()[0].text().split('(')[-1].strip(')')
        tfm_history_chunks = self.tfm_history_spinbox.value()
        
        if not self.load_model() and "Uncompressed" not in model_type_str and "Codec" in model_type_str:
            self.log("ERROR: Model could not be loaded. Aborting stream."); return

        self.log(f"--- Starting stream to {target_ip} with {model_type_str} ---")
        
        # Adjust log for DAC's 20ms frame size
        latency_ms = 20 if "DAC" in model_type_str else HOP_SIZE/16000*1000
        self.log(f"Frame size: {CHUNK} samples | Streaming Latency: {latency_ms:.0f} ms")
        
        self.streamer_thread = QThread()
        self.streamer_worker = StreamerWorker(target_ip, self.model, model_type_str, tfm_history_chunks)
        self.streamer_worker.moveToThread(self.streamer_thread)
        self.streamer_worker.log_message.connect(self.log)
        self.streamer_thread.started.connect(self.streamer_worker.run)
        self.streamer_thread.start()
        
        self.connect_button.setEnabled(False); self.disconnect_button.setEnabled(True)
        self.play_file_button.setEnabled(True); self.model_type_combo.setEnabled(False)
        self.status_label.setText("<b>Status:</b> <font color='green'>Connected</font>")

    def stop_streaming(self):
        if self.streamer_worker: self.streamer_worker.stop()
        if self.streamer_thread: self.streamer_thread.quit(); self.streamer_thread.wait()
        self.streamer_worker = None; self.streamer_thread = None
        self.connect_button.setEnabled(True); self.disconnect_button.setEnabled(False)
        self.play_file_button.setEnabled(False); self.model_type_combo.setEnabled(True)
        self.log("Streaming stopped.")
        self.status_label.setText("<b>Status:</b> <font color='red'>Disconnected</font>")

    def closeEvent(self):
        self.stop_streaming()
        if hasattr(self, 'discovery_worker'): self.discovery_worker.stop()
        if hasattr(self, 'discovery_thread'): self.discovery_thread.quit(); self.discovery_thread.wait()
