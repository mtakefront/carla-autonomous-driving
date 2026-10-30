import carla
import math

from hero import find_hero

client = carla.Client("127.0.0.1", 2000)
client.set_timeout(20.0)

world = client.get_world()
vehicle = find_hero(world)

# Speed (km/h)
velocity = vehicle.get_velocity()
speed = 3.6 * math.sqrt(
    velocity.x**2 + velocity.y**2 + velocity.z**2
)

# Nearest driving lane center
transform = vehicle.get_transform()
waypoint = world.get_map().get_waypoint(
    transform.location,
    project_to_road=True,
    lane_type=carla.LaneType.Driving
)

# Signed lateral offset (meters)
dx = transform.location.x - waypoint.transform.location.x
dy = transform.location.y - waypoint.transform.location.y

yaw = math.radians(waypoint.transform.rotation.yaw)
lane_offset = -math.sin(yaw) * dx + math.cos(yaw) * dy

# Heading error (degrees)
heading_error = (
    transform.rotation.yaw - waypoint.transform.rotation.yaw + 180
) % 360 - 180

print(f"Speed:         {speed:.2f} km/h")
print(f"Lane offset:   {lane_offset:.2f} m")
print(f"Heading error: {heading_error:.2f} degrees")
