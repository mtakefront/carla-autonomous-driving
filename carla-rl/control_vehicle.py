import carla
import time

from hero import find_hero

client = carla.Client("127.0.0.1", 2000)
client.set_timeout(20.0)

world = client.get_world()
vehicle = find_hero(world)

print("Controlling vehicle:", vehicle.id)

try:
    vehicle.apply_control(carla.VehicleControl(
        throttle=0.35,
        steer=0.0,
        brake=0.0
    ))

    print("Moving forward...")
    time.sleep(3)

finally:
    vehicle.apply_control(carla.VehicleControl(
        throttle=0.0,
        steer=0.0,
        brake=1.0
    ))
    print("Braking!")

time.sleep(2)

velocity = vehicle.get_velocity()
speed = 3.6 * (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5

print(f"Final speed: {speed:.2f} km/h")
