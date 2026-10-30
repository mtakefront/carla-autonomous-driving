import carla
import time

client = carla.Client("127.0.0.1", 2000)
client.set_timeout(20.0)

world = client.get_world()
vehicle = world.get_actor(24)

if vehicle is None or vehicle.type_id != "vehicle.tesla.model3":
    raise RuntimeError("Tesla Model 3 with ID 24 not found")

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
