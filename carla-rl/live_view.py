import carla
import io
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from hero import find_hero

client = carla.Client("127.0.0.1", 2000)
client.set_timeout(20.0)
world = client.get_world()

bp = world.get_blueprint_library().find("sensor.camera.rgb")
bp.set_attribute("image_size_x", "800")
bp.set_attribute("image_size_y", "600")
bp.set_attribute("sensor_tick", "0.1")
bp.set_attribute("fov", "90")

try:
    vehicle = find_hero(world)
except RuntimeError:
    vehicle = None

if vehicle is not None:
    camera = world.spawn_actor(
        bp,
        carla.Transform(carla.Location(x=-6, z=3)),
        attach_to=vehicle
    )
    print("Camera attached to vehicle:", vehicle.id)
else:
    spawn = world.get_map().get_spawn_points()[0]
    spawn.location.z += 3
    camera = world.spawn_actor(bp, spawn)
    print("No vehicle found; using fixed camera")

latest = None
lock = threading.Lock()
running = True

def on_image(image):
    global latest
    # CARLA can encode its raw BGRA image directly as JPEG.
    # Use Pillow to convert the frame for browser streaming.
    from PIL import Image
    img = Image.frombytes(
        "RGBA", (image.width, image.height),
        bytes(image.raw_data), "raw", "BGRA"
    ).convert("RGB")
    output = io.BytesIO()
    img.save(output, format="JPEG", quality=75)
    with lock:
        latest = output.getvalue()

camera.listen(on_image)

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            page = b"""<!DOCTYPE html>
<html>
<head><title>CARLA Live View</title>
<style>
body {background:#111;color:white;text-align:center;font-family:Arial}
img {width:90%;max-width:1000px}
</style></head>
<body>
<h2>CARLA Live Camera</h2>
<img src="/stream">
</body></html>"""
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header(
                "Content-Type",
                "multipart/x-mixed-replace; boundary=frame"
            )
            self.end_headers()
            try:
                while running:
                    with lock:
                        frame = latest
                    if frame:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(
                            f"Content-Length: {len(frame)}\r\n\r\n".encode()
                        )
                        self.wfile.write(frame + b"\r\n")
                        self.wfile.flush()
                    time.sleep(0.1)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_error(404)

    def log_message(self, *args):
        pass

server = ThreadingHTTPServer(("0.0.0.0", 8080), Handler)

try:
    print("Live view server running on port 8080")
    server.serve_forever()
finally:
    running = False
    server.server_close()
    camera.stop()
    camera.destroy()
