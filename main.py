import argparse
import asyncio
from xmlrpc import client
import cv2
import aiohttp
from aiortc import RTCConfiguration, RTCIceServer ,RTCPeerConnection, RTCSessionDescription, VideoStreamTrack, MediaStreamTrack
from av import VideoFrame, AudioFrame
import numpy as np
import paho.mqtt.client as mqtt
import json 
from aiortc.contrib.media import MediaPlayer
import mmap
import struct
import time 
import os 
import zmq 
from threading import Thread
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import queue
from fractions import Fraction

# Initialize GStreamer
Gst.init(None)
offer_received = asyncio.Event()
connection_closed = asyncio.Event()
offer_data = None 
loop = None 
currently_connected = False
pc = None
current_time = time.time()
audio_output_queue = queue.Queue()
audio_stream = None
mic_input_queue = None  # Will be asyncio.Queue, initialized in async context
mic_stream = None

def detect_video_stop():
    global current_time, connection_closed, currently_connected
    while not connection_closed.is_set():
        if time.time() - current_time > 2:
            send_img_to_zmq(np.zeros((720, 1280, 3), dtype=np.uint8))  # Send a black frame to indicate disconnection
            time.sleep(0.5)  # Wait a bit before sending the next black frame
        else:
            time.sleep(0.1)  # Check every 100ms

# GStreamer audio pipeline
gst_pipeline = None
gst_appsrc = None

def start_audio_output():
    """Start the GStreamer audio pipeline for playback"""
    global gst_pipeline, gst_appsrc
    if gst_pipeline is None:
        try:
            # Create GStreamer pipeline: appsrc -> audioconvert -> autoaudiosink
            # appsrc accepts raw audio data, autoaudiosink plays to system audio
            pipeline_str = (
                "appsrc name=src is-live=true block=false format=GST_FORMAT_TIME "
                "caps=audio/x-raw,format=S16LE,rate=48000,channels=1,layout=interleaved ! "
                "audioconvert ! autoaudiosink"
            )
            gst_pipeline = Gst.parse_launch(pipeline_str)
            gst_appsrc = gst_pipeline.get_by_name("src")
            
            # Start the pipeline
            gst_pipeline.set_state(Gst.State.PLAYING)
            print("GStreamer audio pipeline started (48kHz, mono, S16LE)")
        except Exception as e:
            print(f"Failed to start GStreamer audio pipeline: {e}")
            gst_pipeline = None
            gst_appsrc = None

def stop_audio_output():
    """Stop the GStreamer audio pipeline"""
    global gst_pipeline, gst_appsrc
    if gst_pipeline is not None:
        try:
            gst_pipeline.set_state(Gst.State.NULL)
            print("GStreamer audio pipeline stopped")
        except Exception as e:
            print(f"Error stopping GStreamer audio pipeline: {e}")
        finally:
            gst_pipeline = None
            gst_appsrc = None

def push_audio_to_gstreamer(audio_data):
    """Push audio data to GStreamer pipeline"""
    global gst_appsrc
    if gst_appsrc is None:
        return
    
    try:
        # Convert numpy array to bytes
        audio_bytes = audio_data.tobytes()
        
        # Create GStreamer buffer
        buf = Gst.Buffer.new_allocate(None, len(audio_bytes), None)
        buf.fill(0, audio_bytes)
        
        # Push buffer to appsrc
        result = gst_appsrc.emit("push-buffer", buf)
        if result != Gst.FlowReturn.OK:
            print(f"GStreamer push-buffer failed: {result}")
    except Exception as e:
        print(f"Error pushing audio to GStreamer: {e}")

# GStreamer mic capture pipeline
gst_mic_pipeline = None
gst_mic_appsink = None

def start_mic_input():
    """Start the GStreamer microphone capture pipeline"""
    global gst_mic_pipeline, gst_mic_appsink
    if gst_mic_pipeline is None:
        try:
            # Create GStreamer pipeline: autoaudiosrc -> audioconvert -> audioresample -> appsink
            # autoaudiosrc captures from default microphone
            # appsink allows pulling raw audio buffers
            pipeline_str = (
                "autoaudiosrc ! "
                "audioconvert ! "
                "audioresample ! "
                "audio/x-raw,format=S16LE,rate=48000,channels=1 ! "
                "appsink name=micsink emit-signals=true max-buffers=10 drop=true"
            )
            gst_mic_pipeline = Gst.parse_launch(pipeline_str)
            gst_mic_appsink = gst_mic_pipeline.get_by_name("micsink")
            
            # Start the pipeline
            gst_mic_pipeline.set_state(Gst.State.PLAYING)
            print("GStreamer mic capture pipeline started (48kHz, mono, S16LE)")
        except Exception as e:
            print(f"Failed to start GStreamer mic pipeline: {e}")
            gst_mic_pipeline = None
            gst_mic_appsink = None

def stop_mic_input():
    """Stop the GStreamer microphone capture pipeline"""
    global gst_mic_pipeline, gst_mic_appsink
    if gst_mic_pipeline is not None:
        try:
            gst_mic_pipeline.set_state(Gst.State.NULL)
            print("GStreamer mic capture pipeline stopped")
        except Exception as e:
            print(f"Error stopping GStreamer mic pipeline: {e}")
        finally:
            gst_mic_pipeline = None
            gst_mic_appsink = None

def pull_audio_from_gstreamer():
    """Pull audio data from GStreamer appsink. Returns numpy array or None."""
    global gst_mic_appsink
    if gst_mic_appsink is None:
        return None
    
    try:
        sample = gst_mic_appsink.emit("pull-sample")
        if sample is None:
            return None
        
        buf = sample.get_buffer()
        caps = sample.get_caps()
        struct = caps.get_structure(0)
        
        # Get buffer data as bytes
        success, map_info = buf.map(Gst.MapFlags.READ)
        if not success:
            return None
        
        # Convert bytes to numpy int16 array
        audio_data = np.frombuffer(map_info.data, dtype=np.int16).copy()
        buf.unmap(map_info)
        return audio_data
    except Exception as e:
        print(f"Error pulling audio from GStreamer: {e}")
        return None

class MicrophoneAudioTrack(MediaStreamTrack):
    """Audio track that captures from local microphone via GStreamer"""
    kind = "audio"
    
    def __init__(self):
        super().__init__()
        self._timestamp = 0
        self.sample_rate = 48000
        self.samples_per_frame = 480  # 20ms at 48kHz
        start_mic_input()
    
    async def recv(self):
        # Pull audio data from GStreamer appsink
        try:
            # Use run_in_executor to avoid blocking the event loop
            audio_data = await asyncio.get_event_loop().run_in_executor(
                None, pull_audio_from_gstreamer
            )
            
            if audio_data is None:
                # No data available, generate silence
                audio_data = np.zeros(self.samples_per_frame, dtype=np.int16)
            
            # Ensure correct shape and size
            if len(audio_data) < self.samples_per_frame:
                # Pad with zeros
                padded = np.zeros(self.samples_per_frame, dtype=np.int16)
                padded[:len(audio_data)] = audio_data
                audio_data = padded
            elif len(audio_data) > self.samples_per_frame:
                # Truncate
                audio_data = audio_data[:self.samples_per_frame]
            
            # Apply volume boost (10x) for microphone - be careful not to clip
            audio_data = np.clip(audio_data.astype(np.int32) * 10, -32768, 32767).astype(np.int16)
            
            # Reshape to (1, samples) for AudioFrame - mono layout
            audio_array = audio_data.reshape(1, -1)
            
            # Create AudioFrame with mono layout
            frame = AudioFrame.from_ndarray(
                audio_array,
                format='s16',
                layout='mono'
            )
            frame.sample_rate = self.sample_rate
            frame.pts = self._timestamp
            frame.time_base = Fraction(1, 48000)
            
            self._timestamp += self.samples_per_frame
            
            return frame
            
        except Exception as e:
            print(f"Error in microphone recv: {e}")
            # Return silence on error
            audio_array = np.zeros((1, self.samples_per_frame), dtype=np.int16)
            frame = AudioFrame.from_ndarray(
                audio_array,
                format='s16',
                layout='mono'
            )
            frame.sample_rate = self.sample_rate
            frame.pts = self._timestamp
            frame.time_base = Fraction(1, 48000)
            self._timestamp += self.samples_per_frame
            return frame
    
    def stop(self):
        super().stop()
        stop_mic_input()

async def consume_audio_track(track):
    """Consume audio track and play through GStreamer"""
    global connection_closed
    print(f"Started receiving remote audio track: {track.kind}")
    
    start_audio_output()
    frame_count = 0
    
    try:
        while not connection_closed.is_set():
            try:
                frame = await asyncio.wait_for(track.recv(), timeout=1.0)
                
                # Convert audio frame to numpy array
                # audio_array shape is (channels, samples)
                audio_array = frame.to_ndarray()
                
                # Log detailed info for first few frames
                if frame_count < 5:
                    print(f"Audio frame {frame_count}: shape={audio_array.shape}, sample_rate={frame.sample_rate}, pts={frame.pts}")
                    frame_count += 1
                
                # Frame has 1920 samples but pts increments by 960 (20ms at 48kHz)
                # Resample 1920 -> 960 to match the correct duration
                audio_flat = audio_array.flatten().astype(np.float32)
                
                # Resample from 1920 to 960 samples using linear interpolation
                indices = np.linspace(0, len(audio_flat) - 1, 960)
                audio_resampled = np.interp(indices, np.arange(len(audio_flat)), audio_flat)
                
                # Convert back to int16 for GStreamer
                audio_data = np.clip(audio_resampled, -32768, 32767).astype(np.int16)
                
                # Push directly to GStreamer pipeline
                push_audio_to_gstreamer(audio_data)
                
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error receiving audio frame: {e}")
                import traceback
                traceback.print_exc()
                break
                
    except asyncio.CancelledError:
        print("Audio track consumption stopped.")
    finally:
        stop_audio_output()
            
class DummyVideoTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        self.counter = 0

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        x = int(320 + 200 * np.sin(self.counter / 10))
        y = int(240 + 150 * np.cos(self.counter / 10))
        cv2.circle(img, (x, y), 30, (0, 255, 0), -1)
        cv2.putText(img, f"Sending Frame: {self.counter}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        self.counter += 1
        frame = VideoFrame.from_ndarray(img, format="bgr24")
        frame.pts = pts
        frame.time_base = time_base
        return frame

class RealVideoTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        self.counter = 0
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.dummy_frame = np.zeros((480,640,3), dtype=np.uint8)
        if not self.cap.isOpened():
            raise IOError("Could not open camera")
    
    async def recv(self):
        pts, time_base = await self.next_timestamp()
        ret, frame = self.cap.read()
        if ret:
            v_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        else:
            v_frame = VideoFrame.from_ndarray(self.dummy_frame, format="bgr24")
        self.counter += 1
        v_frame.pts = pts
        v_frame.time_base = time_base
        return v_frame


def send_img_to_zmq(img):
    rsz_img = cv2.resize(img, (1280, 720))
    height, width, channels = rsz_img.shape
    raw_bytes = rsz_img.tobytes()
    header = f"{width},{height},{channels}".encode('utf-8')
    socket.send_multipart([header, raw_bytes])

async def consume_track(track):
    global current_time, connection_closed
    # global shm_img
    connection_closed.clear()  # Reset the event at the start of track consumption
    print(f"Started receiving remote track: {track.kind}")
    thread = Thread(target=detect_video_stop, daemon=True)
    thread.start()
    try:
        while True:
            frame = await track.recv()
            img = frame.to_ndarray(format="bgr24")
            send_img_to_zmq(img)
            current_time = time.time()  # Update the last received time
    except asyncio.CancelledError:
        print("Track consumption stopped.")
    except Exception as e:
        print(e)
    finally:
        cv2.destroyAllWindows()

def setup_peer_connection():
    global is_host
    config = RTCConfiguration(
        iceServers=[
            # Public Google STUN servers
            RTCIceServer(urls="stun:stun.l.google.com:19302"),
            RTCIceServer(urls="stun:stun1.l.google.com:19302"),
            RTCIceServer(urls="stun:stun2.l.google.com:19302"),
            RTCIceServer(urls="stun:stun3.l.google.com:19302"),
            RTCIceServer(urls="stun:stun4.l.google.com:19302"),
            # Free public TURN servers for carrier-grade NAT (mobile data)
            # OpenRelay - free public TURN
            RTCIceServer(
                urls="turn:openrelay.metered.ca:80?transport=tcp",
                username="openrelayproject",
                credential="openrelayproject"
            ),
            RTCIceServer(
                urls="turn:openrelay.metered.ca:443?transport=tcp",
                username="openrelayproject",
                credential="openrelayproject"
            ),
            # FreeTURN - another free public TURN service
            RTCIceServer(
                urls="turn:freeturn.net:3478",
                username="free",
                credential="free"
            ),
            # Public TURN from turn.geforcenow.com (NVIDIA)
            RTCIceServer(
                urls="turn:turn.geforcenow.com:3478",
                username="nvidia",
                credential="nvidia"
            ),
        ]
    )
    pc = RTCPeerConnection(configuration=config)
    local_video_track = DummyVideoTrack()
    local_audio_track = MicrophoneAudioTrack()
    pc.addTrack(local_video_track)
    pc.addTrack(local_audio_track)
    print("Added local video and microphone audio tracks to peer connection")
    
    @pc.on("track")
    def on_track(track):
        print(f"Received track: {track.kind}")
        if track.kind == "video":
            asyncio.create_task(consume_track(track))
        elif track.kind == "audio":
            asyncio.create_task(consume_audio_track(track))
    def on_connectionstatechange():
        global currently_connected
        print(f"Connection state changed: {pc.connectionState}")
        if pc.connectionState == "connected":
            currently_connected = True
        elif pc.connectionState in ("disconnected", "failed", "closed"):
            currently_connected = False
        if pc.connectionState == "failed":
            asyncio.create_task(pc.close())
    pc.on("connectionstatechange", on_connectionstatechange)
    return pc

# --- ROLE 1: HOST (SERVER) ---
async def run_host():
    global pc, loop, offer_data, offer_received, mqttc, currently_connected, connection_closed
    loop = asyncio.get_event_loop()

    while True:
        pc = setup_peer_connection()
        offer_received.clear()
        offer_data = None
        
        print("Waiting for offer via MQTT...")
        try:
            await offer_received.wait()
            
            if offer_data is None:
                print("Error: No valid offer data received, restarting...")
                await pc.close()
                continue
            print("Offer received, setting remote description...")
            # Print SDP to see audio codec and sample rate info
            print("=" * 50)
            print("OFFER SDP:")
            print(offer_data.sdp)
            print("=" * 50)
            
            # This now sets the remote description containing ALL cellular/TURN endpoints instantly
            print(f"Setting remote description with type: {offer_data.type}")
            await pc.setRemoteDescription(offer_data)
            print("Remote description set successfully")
            
            print("Creating answer...")
            answer = await pc.createAnswer()
            print(f"Answer created, setting local description...")
            await pc.setLocalDescription(answer)
            print(f"Local description set. SDP type: {pc.localDescription.type}")
            
            # Wait for Python to gather its own STUN/TURN candidates before publishing back.
            # Timeout after 15s so we don't hang forever if STUN/TURN servers are unreachable.
            print("Gathering Python ICE candidates...")
            gather_timeout = 15.0
            elapsed = 0.0
            while pc.iceGatheringState != "complete" and elapsed < gather_timeout:
                await asyncio.sleep(0.1)
                elapsed += 0.1
            if pc.iceGatheringState != "complete":
                print(f"WARNING: ICE gathering timed out after {gather_timeout}s — publishing partial candidates anyway")
            else:
                print("ICE gathering complete.")
                
            print("Publishing answer...")
            mqttc.publish("webrtc", json.dumps({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}))

            # Wait until connection fails or is closed
            connection_done = asyncio.Event()
            connecting_start_time = None
            CONNECTING_TIMEOUT = 60  # seconds
            
            def on_state_change():
                nonlocal connecting_start_time
                state = pc.connectionState
                print(f"Connection state: {state}")
                if state == "connecting":
                    if connecting_start_time is None:
                        connecting_start_time = time.time()
                        print("Started tracking 'connecting' state...")
                else:
                    connecting_start_time = None  # Reset when state changes
                    
                if state in ("failed", "closed", "disconnected"):
                    connection_done.set()
            
            pc.on("connectionstatechange", on_state_change)
            
            # Monitor for connecting timeout
            async def check_connecting_timeout():
                nonlocal connecting_start_time
                while not connection_done.is_set():
                    if connecting_start_time is not None:
                        elapsed = time.time() - connecting_start_time
                        if elapsed > CONNECTING_TIMEOUT:
                            print(f"Connection stuck in 'connecting' state for {elapsed:.1f}s (>{CONNECTING_TIMEOUT}s), forcing disconnect...")
                            connection_done.set()
                            asyncio.create_task(pc.close())
                            return
                    await asyncio.sleep(1)
            
            timeout_task = asyncio.create_task(check_connecting_timeout())
            
            # Wait for connection to end
            await connection_done.wait()
            timeout_task.cancel()
            print("Connection ended, closing peer connection and restarting...")
            send_img_to_zmq(np.zeros((720, 1280, 3), dtype=np.uint8))  # Send a black frame to indicate disconnection
            connection_closed.set()  # Signal the video stop detection thread to exit
        except asyncio.CancelledError:
            print("Shutting down...")
            await pc.close()
            break
        except Exception as e:
            print(f"Error during connection: {e}")
        finally:
            if pc.connectionState != "closed":
                await pc.close()
        
        # Brief pause before restarting
        await asyncio.sleep(1)


def on_connect(client, userdata, flags, reason_code, properties):
    print(f"Connected to MQTT broker with result code {reason_code}")
    client.subscribe("control")
    
def on_message(client, userdata, msg):
    global pc, offer_data, offer_received, loop, currently_connected
    try:
        if currently_connected:
            return
        
        msg_payload = json.loads(msg.payload.decode("utf-8"))
        if "content" in msg_payload:
            content = msg_payload["content"]
            print(content)
            # We only look for the bundled SDP offer
            if "sdp" in content and "type" in content:
                offer_data = RTCSessionDescription(sdp=content["sdp"], type=content["type"])
                if loop is not None:
                    loop.call_soon_threadsafe(offer_received.set)
        else:
            print(f"Received unexpected message: {msg_payload}")
    except json.JSONDecodeError:
        print("Error decoding JSON message")
    # FIXED: Removed trailing () from offer_received.set
    
if __name__ == "__main__":    
    try:
        global mqttc
        global socket
        socket = zmq.Context().socket(zmq.PUSH)
        endpoint = "ipc:///tmp/zmq_pubsub.ipc"
        socket.bind(endpoint)
        mqttc = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        mqttc.on_connect = on_connect
        mqttc.on_message = on_message

        # Ensure your broker IP and port are reachable
        mqttc.connect("162.254.35.41", 6789, 60)
        mqttc.loop_start()
        
        asyncio.run(run_host())
    except KeyboardInterrupt:
        print("Interrupted by user, shutting down...")
    except Exception as e:
        print(f"Unexpected error: {e}")
