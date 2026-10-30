import math
import time

import carla
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from hero import find_hero


ROUTE_SPACING = 1.0            # meters between route waypoints
LOOKAHEAD = (5, 10, 20, 40)    # meters ahead for future headings


class CarlaEnv(gym.Env):
    def __init__(self, vehicle_id=None, max_steps=3000, route_length=2000):
        super().__init__()

        self.client = carla.Client("127.0.0.1", 2000)
        self.client.set_timeout(20.0)
        self.world = self.client.get_world()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)
        # Default: the hero vehicle; vehicle_id picks a specific actor
        if vehicle_id is None:
            self.vehicle = find_hero(self.world)
        else:
            self.vehicle = self.world.get_actor(vehicle_id)

        if self.vehicle is None:
            raise RuntimeError("Vehicle not found")

        self.start_transform = self.vehicle.get_transform()
        self.map = self.world.get_map()
        self.start_lane = self.map.get_waypoint(
            self.start_transform.location
        )
        self.steps = 0
        self.stuck_steps = 0
        self.max_steps = max_steps
        self.prev_steer = 0.0

        # Fixed reference route: offset/heading are measured against this
        # lane, so moving to a neighbouring lane is penalized and junctions
        # are unambiguous
        self.route = self._build_route(route_length)
        self.route_xy = np.array(
            [[wp.transform.location.x, wp.transform.location.y]
             for wp in self.route],
            dtype=np.float64,
        )
        self.route_idx = 0

        # Remove collision sensors left on this vehicle by a killed run
        for sensor in self.world.get_actors().filter("sensor.other.collision"):
            if sensor.parent is not None and sensor.parent.id == self.vehicle.id:
                sensor.destroy()

        # Collision sensor: sets a flag, step() terminates on it
        self.collided = False
        blueprint = self.world.get_blueprint_library().find(
            "sensor.other.collision"
        )
        self.collision_sensor = self.world.spawn_actor(
            blueprint, carla.Transform(), attach_to=self.vehicle
        )
        self.collision_sensor.listen(self._on_collision)

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
        )

        # speed, offset, heading, 4 future headings, previous steer
        self.observation_space = spaces.Box(
            low=np.array([0.0, -10.0, -180.0] + [-180.0] * len(LOOKAHEAD) + [-1.0],
                         dtype=np.float32),
            high=np.array([100.0, 10.0, 180.0] + [180.0] * len(LOOKAHEAD) + [1.0],
                          dtype=np.float32),
        )

    def _build_route(self, length):
        # Follow the start lane; at branches take the one with the least
        # heading change (same choice the old lookahead made)
        route = [self.start_lane]
        wp = self.start_lane
        for _ in range(int(length / ROUTE_SPACING)):
            candidates = wp.next(ROUTE_SPACING)
            if not candidates:
                break
            wp = min(
                candidates,
                key=lambda c: abs(
                    (c.transform.rotation.yaw
                     - wp.transform.rotation.yaw
                     + 180) % 360 - 180
                )
            )
            route.append(wp)
        return route

    def _on_collision(self, event):
        self.collided = True

    def _get_state(self):
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()

        speed = 3.6 * math.sqrt(
            velocity.x**2 + velocity.y**2 + velocity.z**2
        )

        # Nearest route point, searched only around the current progress so
        # the route crossing itself does not cause a jump
        lo = max(0, self.route_idx - 10)
        hi = min(len(self.route), self.route_idx + 60)
        loc = np.array([transform.location.x, transform.location.y])
        dists = np.sum((self.route_xy[lo:hi] - loc) ** 2, axis=1)
        self.route_idx = lo + int(np.argmin(dists))
        ref = self.route[self.route_idx]

        dx = transform.location.x - ref.transform.location.x
        dy = transform.location.y - ref.transform.location.y
        yaw = math.radians(ref.transform.rotation.yaw)

        offset = -math.sin(yaw) * dx + math.cos(yaw) * dy
        heading = (
            transform.rotation.yaw - ref.transform.rotation.yaw + 180
        ) % 360 - 180

        # Off-road only if no driving lane exists under the vehicle; crossing
        # a lane line (e.g. cutting a corner in a turn) does not count
        on_road = self.map.get_waypoint(
            transform.location,
            project_to_road=False,
            lane_type=carla.LaneType.Driving,
        ) is not None

        future_headings = []

        for distance in LOOKAHEAD:
            idx = min(
                self.route_idx + int(distance / ROUTE_SPACING),
                len(self.route) - 1,
            )
            future_heading = (
                self.route[idx].transform.rotation.yaw
                - transform.rotation.yaw
                + 180
            ) % 360 - 180
            future_headings.append(future_heading)

        obs = np.array(
            [speed, offset, heading, *future_headings, self.prev_steer],
            dtype=np.float32
        )
        return obs, on_road

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.vehicle.apply_control(
            carla.VehicleControl(throttle=0.0, brake=1.0)
        )
        self.vehicle.set_simulate_physics(False)
        self.vehicle.set_transform(self.start_transform)
        self.vehicle.set_target_velocity(carla.Vector3D())
        self.vehicle.set_target_angular_velocity(carla.Vector3D())
        self.vehicle.set_simulate_physics(True)

        self.steps = 0
        self.stuck_steps = 0
        self.route_idx = 0
        self.prev_steer = 0.0
        self.world.tick()
        # Ignore any collision event caused by the teleport itself
        self.collided = False

        obs, _ = self._get_state()
        return obs, {}

    def step(self, action):
        steer = float(np.clip(action[0], -1.0, 1.0))
        longitudinal = float(np.clip(action[1], -1.0, 1.0))

        # Small negative values coast instead of braking, so a random
        # policy does not spend half its time braking
        throttle = max(0.0, longitudinal)
        brake = max(0.0, -longitudinal - 0.3) / 0.7

        self.vehicle.apply_control(
            carla.VehicleControl(
                steer=steer,
                throttle=throttle,
                brake=brake,
            )
        )

        self.world.tick()

        self.steps += 1

        steer_change = abs(steer - self.prev_steer)
        self.prev_steer = steer

        obs, on_road = self._get_state()
        speed, offset, heading = obs[:3]

        if speed < 2.0:
            self.stuck_steps += 1
        else:
            self.stuck_steps = 0

        # Only speed along the route counts, so driving backwards or
        # sideways is not rewarded. All terms are roughly in [-1, 1]
        forward_speed = speed * math.cos(math.radians(heading))
        speed_reward = 1.0 - abs(forward_speed - 40.0) / 40.0   # +1 at 40 km/h
        lane_penalty = min((offset / 1.75) ** 2, 2.0)  # neighbour lane ~ -1
        heading_penalty = min(abs(heading) / 90.0, 1.0)
        steer_penalty = steer_change                    # discourages zigzag

        reward = (
            speed_reward
            - 0.5 * lane_penalty
            - 0.3 * heading_penalty
            - 0.2 * steer_penalty
        )

        off_road = not on_road
        lost_route = abs(offset) > 6.0     # too far for route tracking
        stuck = self.steps > 40 and self.stuck_steps >= 40   # 2 s at 20 Hz

        terminated = bool(off_road or lost_route or stuck or self.collided)
        # Large enough that ending the episode is never better than
        # continuing from a bad state
        if terminated:
            reward = -30.0

        route_end = self.route_idx >= len(self.route) - 1 - LOOKAHEAD[-1]
        truncated = self.steps >= self.max_steps or route_end

        info = {
            "off_road": off_road,
            "lost_route": lost_route,
            "stuck": stuck,
            "collision": self.collided,
            "route_end": route_end,
        }
        return obs, float(reward), terminated, truncated, info

    def close(self):
        if self.collision_sensor is not None:
            self.collision_sensor.stop()
            self.collision_sensor.destroy()
            self.collision_sensor = None
        self.vehicle.apply_control(
            carla.VehicleControl(throttle=0.0, brake=1.0)
        )
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        self.world.apply_settings(settings)
