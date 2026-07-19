import argparse
import asyncio
from xmlrpc import client
import cv2
import aiohttp
from aiortc import RTCConfiguration, RTCIceServer ,RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame
import numpy as np
import paho.mqtt.client as mqtt
import json 
from aiortc.contrib.media import MediaPlayer
import mmap
import struct
import time 
import os 
import zmq 

offer_received = asyncio.Event()
connection_closed = asyncio.Event()
offer_data = None 
loop = None 
currently_connected = False
pc = None

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

async def consume_track(track):
    # global shm_img
    print(f"Started receiving remote track: {track.kind}")
    try:
        while True:
            frame = await track.recv()
            img = frame.to_ndarray(format="bgr24")
            rsz_img = cv2.resize(img, (1280, 720))
            height, width, channels = rsz_img.shape
            raw_bytes = rsz_img.tobytes()
            # Send metadata first (or wrap it in a multipart message)
            # For simplicity, send a multipart message: [header, raw_bytes]
            header = f"{width},{height},{channels}".encode('utf-8')
            socket.send_multipart([header, raw_bytes])
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
    local_track = DummyVideoTrack()
    pc.addTrack(local_track)
    
    @pc.on("track")
    def on_track(track):
        if track.kind == "video":
            asyncio.create_task(consume_track(track))
        elif track.kind == "audio":
            player = MediaPlayer("default", format="pulse")
        
            player.addTrack(track)
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
    global pc, loop, offer_data, offer_received, mqttc, currently_connected
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
            # REMOVED: await iceReceived.wait() 
            
            # This now sets the remote description containing ALL cellular/TURN endpoints instantly
            await pc.setRemoteDescription(offer_data)
            
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
            
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
            
            def on_state_change():
                state = pc.connectionState
                print(f"Connection state: {state}")
                if state in ("failed", "closed", "disconnected"):
                    connection_done.set()
            
            pc.on("connectionstatechange", on_state_change)
            
            # Wait for connection to end
            await connection_done.wait()
            print("Connection ended, closing peer connection and restarting...")
            
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
