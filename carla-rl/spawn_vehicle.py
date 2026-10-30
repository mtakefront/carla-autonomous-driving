import carla

client = carla.Client("127.0.0.1", 2000)
client.set_timeout(20.0)

world = client.get_world()

blueprint = world.get_blueprint_library().find("vehicle.tesla.model3")
spawn_points = world.get_map().get_spawn_points()

vehicle = None

for spawn_point in spawn_points:
    vehicle = world.try_spawn_actor(blueprint, spawn_point)
    if vehicle is not None:
        break

if vehicle is None:
    raise RuntimeError("No available spawn point")

vehicle.set_autopilot(False)
vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))

print("Vehicle created!")
print("Vehicle ID:", vehicle.id)
print("Location:", vehicle.get_location())
