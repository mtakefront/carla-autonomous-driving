import os, queue, sys, time
import carla

port = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
client = carla.Client("localhost", port)
client.set_timeout(60.0)
print("client", client.get_client_version(), "| server", client.get_server_version())

world = client.get_world()
print("map:", world.get_map().name)

settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)

bp_lib = world.get_blueprint_library()
vehicle = world.spawn_actor(bp_lib.filter("vehicle.tesla.model3")[0],
                            world.get_map().get_spawn_points()[0])
cam_bp = bp_lib.find("sensor.camera.rgb")
cam_bp.set_attribute("image_size_x", "800")
cam_bp.set_attribute("image_size_y", "600")
camera = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=-6, z=3), carla.Rotation(pitch=-15)),
                           attach_to=vehicle)
images = queue.Queue()
camera.listen(images.put)
vehicle.set_autopilot(True, client.get_trafficmanager(8000).get_port())

t0 = time.time()
for _ in range(100):
    world.tick()
img = None
while not images.empty():
    img = images.get()
print("100 ticks in %.1fs (%.0f FPS) | last frame %d" % (time.time() - t0, 100 / (time.time() - t0), img.frame))
print("vehicle at", vehicle.get_location())
frame_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_frame.png")
img.save_to_disk(frame_path)
print("saved", frame_path)

camera.stop(); camera.destroy(); vehicle.destroy()
settings.synchronous_mode = False
world.apply_settings(settings)
print("OK")
