import carla

client = carla.Client("127.0.0.1", 2000)
client.set_timeout(20.0)

world = client.get_world()
vehicle = world.get_actor(24)

if vehicle is None:
    raise RuntimeError("Vehicle 24 not found")

waypoint = world.get_map().get_waypoint(
    vehicle.get_location(),
    project_to_road=True,
    lane_type=carla.LaneType.Driving,
)

vehicle.apply_control(carla.VehicleControl(throttle=0, brake=1))
vehicle.set_simulate_physics(False)
vehicle.set_transform(waypoint.transform)
vehicle.set_target_velocity(carla.Vector3D())
vehicle.set_target_angular_velocity(carla.Vector3D())
vehicle.set_simulate_physics(True)

print("Vehicle aligned with lane.")
