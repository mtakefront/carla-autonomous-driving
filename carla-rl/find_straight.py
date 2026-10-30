import carla

client = carla.Client("127.0.0.1", 2000)
client.set_timeout(20)

world = client.get_world()
carla_map = world.get_map()

for start in carla_map.get_spawn_points():
    wp = carla_map.get_waypoint(start.location)

    current = wp
    valid = True

    for _ in range(40):
        next_points = current.next(1.0)

        if len(next_points) != 1:
            valid = False
            break

        current = next_points[0]

        yaw_diff = (
            current.transform.rotation.yaw
            - wp.transform.rotation.yaw + 180
        ) % 360 - 180

        if (
            abs(yaw_diff) > 3.0
            or current.road_id != wp.road_id
            or current.lane_id != wp.lane_id
        ):
            valid = False
            break

    if valid:
        vehicle = world.get_actor(24)
        if vehicle is None:
            raise RuntimeError("Vehicle 24 not found")

        vehicle.apply_control(carla.VehicleControl(brake=1))
        vehicle.set_simulate_physics(False)
        vehicle.set_transform(start)
        vehicle.set_target_velocity(carla.Vector3D())
        vehicle.set_target_angular_velocity(carla.Vector3D())
        vehicle.set_simulate_physics(True)

        print("Found straight road (40 m)")
        print("Road ID:", wp.road_id)
        print("Lane ID:", wp.lane_id)
        print("Vehicle moved!")
        break
else:
    print("No suitable straight road found")
