"""CARLA 0.9.16 / Gymnasium navigation environment using map-based virtual perception.

Requires CARLA PythonAPI's `agents` directory on PYTHONPATH and existing `hero.py`.
Single synchronous CARLA world owner; do not run another world.tick() client concurrently.
"""

import bisect
import math
from collections import deque

import carla
import gymnasium as gym
import networkx as nx
import numpy as np
from gymnasium import spaces

from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.local_planner import RoadOption
from hero import find_hero


ROUTE_SPACING = 2.0
START_CLEARANCE = 10.0  # meters of non-junction road required at the start
# Dense enough to show where a junction arc starts bending before the car
# reaches it (with only 20/40 m, left turns looked straight until mid-junction);
# 50 m so turns and lights ahead can be prepared for early.
LOOKAHEAD = (0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0)
LOOKAHEAD_STEP = 2.5  # walk step; must divide every LOOKAHEAD distance
MAX_LANES = 6
GOAL_SPACING = 10.0  # meters between candidate goal points on driving lanes
JUNCTION_ATTEMPT_M = 10.0  # failing this close before a maneuver counts against it
# Traffic lights and stop signs (OpenDRIVE signal types)
LIGHT_TYPE, STOP_TYPE = "1000001", "206"
CONTROL_VIEW_M = 50.0  # lights/signs on the ego lane are seen this far ahead
LIGHT_WAIT_M = 30.0    # waiting at red/yellow this close to the line is free
STOP_ZONE_M = 6.0      # a stop counts only within this distance of the line
STOP_SPEED_KMH = 1.0   # "stopped" means below this speed
STOP_WAIT_STEPS = 30   # after stopping at a sign, 3 s of free waiting
CONTROL_KINDS = ("red", "yellow", "green", "stop")
# Traffic and pedestrians
TM_PORT = 8000
FRONT_RANGE_M = 40.0      # the front sensor sees this far along the path
PATH_HALF_WIDTH_M = 2.0   # an object this close to the path is "in front"
QUEUE_GAP_M = 12.0        # waiting behind a moving car or person this close is free
START_FREE_M = 8.0        # no vehicle or pedestrian this close to a start point
START_REAR_FREE_M = 30.0  # nor a car this far behind it in its lane (it couldn't stop)
SIDE_RANGE_M = 25.0       # side/rear sensors: this far ahead and behind
LANE_PENALTY = 0.1        # per 10 Hz step, per meter off center and per rad of heading
MIN_SPEED_KMH = 10.0  # below this, a per-step penalty growing as the car slows
LOW_SPEED_PENALTY = 0.2  # per 20 Hz-step at standstill (x step_scale -> 0.4 at 10 Hz)
LANE_GAP_M = 0.06  # widest no-lane gap measured between adjacent lanes is ~5 cm
MARKING_TYPES = {
    "NONE": 0, "Other": 1, "Broken": 2, "Solid": 3,
    "SolidSolid": 4, "SolidBroken": 5, "BrokenSolid": 6,
    "BrokenBroken": 7, "BottsDots": 8, "Grass": 9, "Curb": 10,
}
VIRTUAL_MARKING = 11  # edge of the route's path through a junction (no paint)
TURN_OPTIONS = (RoadOption.LEFT, RoadOption.RIGHT, RoadOption.STRAIGHT)


class CarlaEnv(gym.Env):
    """Actions: [steering, combined throttle/brake], each in [-1, 1].
    Longitudinal 0 means light throttle (0.25); coasting from -0.33,
    braking below -0.6.

    Observations (75 floats):
      speed_kmh, speed_limit_kmh (0 = unknown), previous_steer,
      next_maneuver_onehot(LEFT, RIGHT, STRAIGHT) (all 0 = no maneuver),
      distance_to_maneuver_m (100 = none/far), lanes_to_left, lanes_to_right,
      control_onehot(red, yellow, green, stop) (all 0 = none ahead),
      distance_to_stop_line_m (100 = none), stopped_at_this_sign,
      front_object_present, front_distance_m (40 = none), front_closing_speed_mps
          (nearest vehicle/pedestrian on the path ahead, parked cars included),
      lane_offset_m (+ right of center), lane_heading_deg (+ pointing right)
          (own lane; 0 in junctions or off the lane),
      for [left lane, right lane, own lane behind]: [present, longitudinal_m,
          relative_speed_mps] of the nearest vehicle within 25 m (0 in junctions),
      at d=0 for each side [left,right]: [line_y_vehicle_m, marking_type_code]
      for d in [5,10,15,20,30,40,50] and each side: [line_x_vehicle_m, line_y_vehicle_m,
          marking_type_code]

    Vehicle frame: +x forward, +y right. Missing lines are all zeros
    (marking code 0 = no line).
    Lane lines show the ego lane up to the next junction and, past it, the
    road the route continues on (see _lane_ahead); inside junctions, the
    edges of the route's connector as virtual lines (marking code 11).
    Route and waypoint indices are intentionally not in the observation.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        vehicle_id=None,
        max_steps=1500,                # 150 seconds at 10 Hz
        route_length=600.0,            # maximum path length, meters
        min_route_length=120.0,
        max_route_attempts=100,
        fixed_delta_seconds=0.1,       # 10 Hz control
        stuck_timeout_steps=100,       # 10 seconds at 10 Hz
        traffic_vehicles=0,            # autopilot cars (Traffic Manager)
        pedestrians=0,                 # AI walkers
        no_rendering=False,            # faster; nothing is drawn (no Live View)
    ):
        super().__init__()
        self.client = carla.Client("127.0.0.1", 2000)
        self.client.set_timeout(20.0)
        self.world = self.client.get_world()
        self.original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = float(fixed_delta_seconds)
        # Physics substeps must cover the whole tick (0.01 s * 10 = 0.1 s).
        settings.substepping = True
        settings.max_substep_delta_time = 0.01
        settings.max_substeps = max(1, math.ceil(fixed_delta_seconds / 0.01))
        # The env uses no camera, so drawing frames is only for people watching.
        settings.no_rendering_mode = bool(no_rendering)
        self.world.apply_settings(settings)
        # Per-step time terms were tuned at 20 Hz; keep them per-second equal.
        self.step_scale = float(fixed_delta_seconds) / 0.05

        self.vehicle = (
            find_hero(self.world) if vehicle_id is None
            else self.world.get_actor(vehicle_id)
        )
        if self.vehicle is None:
            raise RuntimeError("Vehicle not found")

        self.map = self.world.get_map()
        self.spawn_points = self.map.get_spawn_points()
        if not self.spawn_points:
            raise RuntimeError("Map has no spawn points")
        # Starts stay on spawn points (safe to place a car); goals can be any
        # plain driving-lane point, so routes vary far more than spawn pairs.
        self.goal_candidates = [
            wp for wp in self.map.generate_waypoints(GOAL_SPACING)
            if wp.lane_type == carla.LaneType.Driving
            and wp.lane_id != 0 and not wp.is_junction
        ]
        if not self.goal_candidates:
            raise RuntimeError("Map has no driving-lane goal candidates")

        self.max_steps = int(max_steps)
        self.max_route_length = float(route_length)
        self.min_route_length = float(min_route_length)
        self.max_route_attempts = int(max_route_attempts)
        self.stuck_timeout_steps = int(stuck_timeout_steps)
        if not 0 < self.min_route_length < self.max_route_length:
            raise ValueError("Need 0 < min_route_length < route_length")

        # Build the route graph once; every reset traces a new start/goal pair.
        self.planner = GlobalRoutePlanner(self.map, ROUTE_SPACING)
        self.route = []
        self.route_xy = np.empty((0, 2), dtype=np.float64)
        self.route_s_values = np.array([], dtype=np.float64)
        self.route_keys = []
        self.route_segments = []
        self.turn_events = []
        self.route_total = 0.0
        self.route_s = 0.0
        self.route_idx = 0
        self.route_segment_idx = 0
        self.goal_location = None
        self.start_transform = None
        self.steps = 0
        self.stuck_steps = 0
        self.prev_steer = 0.0
        self._maneuver_idx = 0
        self._episode_parts = {}
        self._light_cache = {}
        self._jpath = None  # path through the junction ahead (see _lane_ahead)
        self._front = None  # (distance, closing speed, moving, its speed m/s)
        self._occupied = []  # vehicle/pedestrian locations at reset
        self._occupied_cars = []
        self._control = None          # (kind, distance, landmark id, light actor)
        self._latched_control = None  # last control passed before the junction
        self._stopped_sign = None     # landmark id of the stop sign stopped at
        self._stop_wait = 0
        self.prev_lane_wp = None
        self.collided = False
        self.episode_start_frame = 0
        self._lane_events = deque()
        self._sensors = []

        # 9 scalar/navigation/lane fields, 6 traffic-control fields, then lane lines: at d=0 only the
        # lateral offset and type (forward offset is always ~0), further
        # ahead forward + lateral offset and type.
        low = ([0.0, 0.0, -1.0] + [0.0] * 3 + [0.0] + [0.0, 0.0]
               + [0.0] * 4 + [0.0, 0.0] + [0.0, 0.0, -30.0]
               + [-3.0, -90.0] + [0.0, -SIDE_RANGE_M, -30.0] * 3)
        high = ([200.0, 200.0, 1.0] + [1.0] * 3 + [100.0] + [float(MAX_LANES)] * 2
                + [1.0] * 4 + [100.0, 1.0] + [1.0, FRONT_RANGE_M, 30.0]
                + [3.0, 90.0] + [1.0, SIDE_RANGE_M, 30.0] * 3)
        for distance in LOOKAHEAD:
            for _side in ("left", "right"):
                if distance == 0.0:
                    low.extend([-60.0, 0.0])
                    high.extend([60.0, float(VIRTUAL_MARKING)])
                else:
                    low.extend([-60.0, -60.0, 0.0])
                    high.extend([60.0, 60.0, float(VIRTUAL_MARKING)])
        self.observation_space = spaces.Box(
            low=np.asarray(low, dtype=np.float32),
            high=np.asarray(high, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Retain the previous script's cleanup of stale sensors attached to hero.
        for sensor in self.world.get_actors():
            if sensor.type_id not in (
                "sensor.other.collision", "sensor.other.lane_invasion"
            ):
                continue
            if sensor.parent is not None and sensor.parent.id == self.vehicle.id:
                sensor.destroy()

        blueprints = self.world.get_blueprint_library()
        collision = self.world.spawn_actor(
            blueprints.find("sensor.other.collision"),
            carla.Transform(), attach_to=self.vehicle,
        )
        collision.listen(self._on_collision)
        self._sensors.append(collision)

        invasion = self.world.spawn_actor(
            blueprints.find("sensor.other.lane_invasion"),
            carla.Transform(), attach_to=self.vehicle,
        )
        invasion.listen(self._on_lane_invasion)
        self._sensors.append(invasion)

        self.traffic_vehicles = int(traffic_vehicles)
        self.pedestrians = int(pedestrians)
        self._npcs, self._walkers, self._walker_ctrls = [], [], []
        self._parked = self._parked_vehicles()
        self._remove_old_traffic()
        self.tm = None
        if self.traffic_vehicles or self.pedestrians:
            self.tm = self.client.get_trafficmanager(TM_PORT)
            self.tm.set_synchronous_mode(True)
            # Cars far from ours skip full physics: same traffic, much cheaper.
            self.tm.set_hybrid_physics_mode(True)
            self.tm.set_hybrid_physics_radius(70.0)
            # Generous gap: the agent can brake hard and without warning.
            self.tm.set_global_distance_to_leading_vehicle(20.0)
            self.world.set_pedestrians_cross_factor(0.1)
        self._spawn_traffic()

    @staticmethod
    def _key(wp):
        return (int(wp.road_id), 1 if wp.lane_id > 0 else -1)

    @staticmethod
    def _distance2d(p, q):
        return math.hypot(p.x - q.x, p.y - q.y)

    @staticmethod
    def _marking_code(mark):
        if mark is None:
            return 0
        name = str(mark.type).split(".")[-1]
        return MARKING_TYPES.get(name, 1)

    def _on_collision(self, event):
        if event.frame >= self.episode_start_frame:
            self.collided = True

    def _on_lane_invasion(self, event):
        if event.frame >= self.episode_start_frame:
            types = tuple(str(mark.type).split(".")[-1]
                          for mark in event.crossed_lane_markings)
            self._lane_events.append((event.frame, types))

    def _sample_route(self):
        """Pick a fresh start/goal and reject unsuitable or broken traces."""
        for _ in range(self.max_route_attempts):
            start = self.spawn_points[int(self.np_random.integers(len(self.spawn_points)))]
            if any(self._distance2d(start.location, loc) < START_FREE_M
                   for loc in self._occupied):
                continue
            # A car coming up behind can't brake for one that appears in front of it.
            if any(-START_REAR_FREE_M < x < 0.0 and abs(y) < 2.0
                   for x, y in (self._to_vehicle_xy(start, loc)
                                for loc in self._occupied_cars)):
                continue
            goal_wp = self.goal_candidates[int(self.np_random.integers(len(self.goal_candidates)))]
            target = goal_wp.transform.location
            straight_dist = self._distance2d(start.location, target)
            if not (0.35 * self.min_route_length <= straight_dist <= self.max_route_length):
                continue
            try:
                # CARLA 0.9.16's planner retains turn-decision state between
                # trace_route() calls; clear it for independent episodes.
                self.planner._previous_decision = RoadOption.VOID
                self.planner._intersection_end_node = -1
                trace = self.planner.trace_route(start.location, target)
            except (nx.NetworkXException, RuntimeError, KeyError, IndexError, ValueError, TypeError):
                continue
            if len(trace) < 3:
                continue
            wps = [wp for wp, _ in trace]
            opts = [opt for _, opt in trace]
            if any(wp.lane_id == 0 or wp.lane_type != carla.LaneType.Driving
                   for wp in wps):
                continue
            points = np.asarray(
                [[wp.transform.location.x, wp.transform.location.y] for wp in wps],
                dtype=np.float64,
            )
            distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
            if np.any(distances > 25.0):  # discontinuous/teleporting trace
                continue
            cumulative = np.r_[0.0, np.cumsum(distances)]
            total = float(cumulative[-1])
            if not (self.min_route_length <= total <= self.max_route_length):
                continue
            if self._distance2d(wps[-1].transform.location, target) > 12.0:
                continue

            keys = [self._key(wp) for wp in wps]
            segments = []
            for idx, key in enumerate(keys):
                if not segments or key != segments[-1]["key"]:
                    segments.append({"key": key, "start": idx, "end": idx,
                                     "junction": wps[idx].is_junction})
                else:
                    segments[-1]["end"] = idx
            if not segments or any(s["key"][1] == 0 for s in segments):
                continue
            # Spawn points in junctions or at road ends can sit on a different
            # road/lane than the trace's first segment; reset() would reject them.
            start_wp = self.map.get_waypoint(
                start.location, project_to_road=False,
                lane_type=carla.LaneType.Driving,
            )
            if start_wp is None or self._key(start_wp) != segments[0]["key"]:
                continue
            # Start on a plain road with room before the first junction, so the
            # car can't slip onto an overlapping connector right after reset.
            first_len = cumulative[segments[0]["end"]] - cumulative[segments[0]["start"]]
            if (start_wp.is_junction or first_len < START_CLEARANCE
                    or any(wp.is_junction for wp, s in zip(wps, cumulative)
                           if s < START_CLEARANCE)):
                continue

            # Consecutive LEFT/RIGHT/STRAIGHT samples make one maneuver event.
            turns = []
            previous = None
            for idx, opt in enumerate(opts):
                if opt in TURN_OPTIONS:
                    if opt != previous:
                        turns.append([idx, idx, opt])  # start, end, option
                    else:
                        turns[-1][1] = idx
                    previous = opt
                else:
                    previous = None

            self.start_transform = start
            # The trace may stop a few meters short of the target; use its
            # end so the 5 m goal radius is always reachable.
            self.goal_location = wps[-1].transform.location
            self.route = wps
            self.route_xy = points
            self.route_s_values = cumulative
            self.route_keys = keys
            self.route_segments = segments
            self.turn_events = turns
            self.route_total = total
            self.route_s = 0.0
            self.route_idx = 0
            self.route_segment_idx = 0
            return

        raise RuntimeError(
            "No valid route found; increase max_route_attempts or change "
            "min_route_length/route_length."
        )

    def _driving_waypoint(self, location):
        """Driving lane under location, or None.

        Town10's OpenDRIVE leaves hairline gaps (median 2 mm, max ~5 cm)
        between adjacent driving lanes, where the map reports no lane at all.
        A point within LANE_GAP_M of a driving lane is on the road.
        """
        wp = self.map.get_waypoint(
            location, project_to_road=False, lane_type=carla.LaneType.Driving
        )
        if wp is not None:
            return wp
        found = []
        for dx, dy in ((LANE_GAP_M, 0.0), (-LANE_GAP_M, 0.0),
                       (0.0, LANE_GAP_M), (0.0, -LANE_GAP_M)):
            wp = self.map.get_waypoint(
                carla.Location(location.x + dx, location.y + dy, location.z),
                project_to_road=False, lane_type=carla.LaneType.Driving,
            )
            if wp is not None:
                found.append(wp)
        # On a line between two lanes, take the lane whose center is nearer,
        # so a gap on the center line can't flip the car into oncoming lanes.
        return min(found, key=lambda w: self._distance2d(w.transform.location, location),
                   default=None)

    def _check_route_status(self, location):
        """Road order + lane direction, without any section_id checks.

        Acceptable: the current segment, the next one, and (if the next one is
        extremely short) the segment after it.
        Inside a junction the map can report any of the overlapping connectors,
        so the car always counts as on the route's connector there; whether it
        took the right turn is judged by the road it exits onto.
        """
        wp = self._driving_waypoint(location)
        result = dict(
            waypoint=wp, on_route=False, wrong_way=False,
            off_road=(wp is None), wrong_road=False, route_key=None,
        )
        if wp is None:
            return result

        road_id, lane_sign = self._key(wp)
        begin = self.route_segment_idx
        end = min(len(self.route_segments), begin + 3)
        candidates = []
        for idx in range(begin, end):
            seg = self.route_segments[idx]
            if idx == begin + 2:
                skipped = self.route_segments[idx - 1]
                skipped_length = (self.route_s_values[skipped["end"]]
                                  - self.route_s_values[skipped["start"]])
                if skipped_length > 3.0:
                    continue
            candidates.append((idx, seg["key"]))

        exact = [i for i, key in candidates
                 if key == (road_id, lane_sign)]
        junction = [i for i, _ in candidates if self.route_segments[i]["junction"]]
        if exact or (wp.is_junction and junction):
            if exact:
                chosen = exact[0]
            else:
                # Near a junction the map may briefly report one of its lanes
                # before the car gets there; only move on to the route's
                # connector once progress has actually reached it.
                connector_start = self.route_s_values[self.route_segments[junction[0]]["start"]]
                chosen = junction[0] if self.route_s >= connector_start - 2.0 else begin
            self.route_segment_idx = max(self.route_segment_idx, chosen)
            result["on_route"] = True
            result["route_key"] = self.route_segments[chosen]["key"]
        elif any(key[0] == road_id for _, key in candidates):
            result["wrong_way"] = True
        elif not wp.is_junction:
            result["wrong_road"] = True
        return result

    def _project_progress(self, location, current_key):
        """Project onto nearby planned polyline edges of the matching road/direction.

        This is for measuring travel distance, not for penalizing lateral offset.
        """
        x, y = location.x, location.y
        lo = max(0, self.route_idx - 15)
        hi = min(len(self.route) - 1, self.route_idx + 65)
        best_d2 = float("inf")
        best_s = None
        for i in range(lo, hi):
            if self.route_keys[i] != current_key:
                continue
            ax, ay = self.route_xy[i]
            bx, by = self.route_xy[i + 1]
            dx, dy = bx - ax, by - ay
            length2 = dx * dx + dy * dy
            t = 0.0 if length2 < 1e-9 else float(np.clip(
                ((x - ax) * dx + (y - ay) * dy) / length2, 0.0, 1.0
            ))
            px, py = ax + t * dx, ay + t * dy
            d2 = (x - px) ** 2 + (y - py) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_s = float(self.route_s_values[i] + t *
                               (self.route_s_values[i + 1] - self.route_s_values[i]))

        # Can occur on a short road that has only one route waypoint.
        if best_s is None:
            for i in range(lo, hi + 1):
                if self.route_keys[i] == current_key:
                    dx, dy = self.route_xy[i, 0] - x, self.route_xy[i, 1] - y
                    d2 = dx * dx + dy * dy
                    if d2 < best_d2:
                        best_d2 = d2
                        best_s = float(self.route_s_values[i])
        return best_s

    def _update_progress(self, status, moved_m):
        if not status["on_route"]:
            return 0.0
        candidate = self._project_progress(
            self.vehicle.get_location(), status["route_key"]
        )
        if candidate is None:
            return 0.0
        # Limit jumps caused by crossing/parallel lanes and sparse route samples.
        delta = float(np.clip(candidate - self.route_s,
                              -moved_m - 0.5, moved_m + 0.5))
        previous = self.route_s
        self.route_s = float(np.clip(self.route_s + delta, 0.0, self.route_total))
        self.route_idx = min(
            len(self.route) - 1,
            max(0, bisect.bisect_right(self.route_s_values, self.route_s) - 1),
        )
        return self.route_s - previous

    def _next_maneuver(self):
        """0=LEFT,1=RIGHT,2=STRAIGHT; valid distinguishes no maneuver.

        A maneuver stays active until the car has passed its last waypoint,
        so the command doesn't vanish halfway through the junction.
        """
        for start, end, option in self.turn_events:
            if self.route_s_values[end] >= self.route_s:
                j = TURN_OPTIONS.index(option)
                cmd = [0.0, 0.0, 0.0]
                cmd[j] = 1.0
                dist = max(0.0, float(self.route_s_values[start] - self.route_s))
                return cmd, min(dist, 100.0), 1.0
        return [0.0, 0.0, 0.0], 100.0, 0.0

    @staticmethod
    def _yaw_change(a, b):
        return abs((b.transform.rotation.yaw - a.transform.rotation.yaw + 180.0)
                   % 360.0 - 180.0)

    def _lane_ahead(self, lane_wp):
        """{distance: waypoint} for each LOOKAHEAD distance.

        Before the next junction: the ego lane (its straightest continuation).
        From the junction on: the path through it that the car can take from
        its own lane to the road the route continues on (see _junction_path),
        then that road. If the car's lane can't make the maneuver, the path
        starts from the nearest lane that can, so it shows up off to the side.
        Junction waypoints are shown as virtual lines.
        """
        ahead = {}
        if lane_wp.is_junction:
            # Keep following the path chosen at the junction entry.
            if self._jpath:
                loc = self.vehicle.get_location()
                i0 = min(range(len(self._jpath)), key=lambda i: self._distance2d(
                    self._jpath[i].transform.location, loc))
                for distance in LOOKAHEAD:
                    i = i0 + int(round(distance / LOOKAHEAD_STEP))
                    ahead[distance] = self._jpath[i] if i < len(self._jpath) else None
                return ahead
            return self._route_ahead()

        ahead[0.0] = lane_wp
        wp, s, entry, junction_s = lane_wp, 0.0, lane_wp, None
        while s < LOOKAHEAD[-1]:
            choices = wp.next(LOOKAHEAD_STEP)
            if not choices:
                break
            entry = wp
            wp = min(choices, key=lambda c: self._yaw_change(wp, c))
            s += LOOKAHEAD_STEP
            if wp.is_junction:
                junction_s = s
                break
            if s in LOOKAHEAD:
                ahead[s] = wp
        if junction_s is None:
            self._jpath = None
            return ahead
        self._jpath = self._junction_path(entry)
        fallback = self._route_ahead()
        for distance in LOOKAHEAD:
            if distance >= junction_s:
                if self._jpath:
                    i = int(round((distance - junction_s) / LOOKAHEAD_STEP))
                    ahead[distance] = self._jpath[i] if i < len(self._jpath) else None
                else:
                    ahead[distance] = fallback[distance]
        return ahead

    def _route_ahead(self):
        """Route waypoints at route_s + distance (fallback when no lane path)."""
        ahead = {}
        for distance in LOOKAHEAD:
            i = bisect.bisect_left(self.route_s_values, self.route_s + distance)
            ahead[distance] = self.route[min(i, len(self.route) - 1)]
        return ahead

    def _junction_path(self, entry):
        """Waypoints (LOOKAHEAD_STEP apart) from the junction entry through a
        connector to the route's next road and on along it.

        Tries the connectors of the car's own lane first, then same-direction
        neighbour lanes by distance; None if none reaches the route's exit.
        """
        exit_key = None
        for i in range(self.route_segment_idx, len(self.route_segments) - 1):
            if self.route_segments[i]["junction"]:
                exit_key = self.route_segments[i + 1]["key"]
                break
        if exit_key is None:
            return None
        lanes, left, right = [entry], entry, entry
        for _ in range(MAX_LANES):
            left = left.get_left_lane() if left is not None else None
            right = right.get_right_lane() if right is not None else None
            for other in (left, right):
                if (other is not None and other.lane_type == carla.LaneType.Driving
                        and other.lane_id * entry.lane_id > 0):
                    lanes.append(other)
        for lane in lanes:
            for first in lane.next(LOOKAHEAD_STEP):
                if not first.is_junction:
                    continue
                path, wp = [first], first
                while wp.is_junction and len(path) < 60:
                    nxt = wp.next(LOOKAHEAD_STEP)
                    if not nxt:
                        break
                    wp = min(nxt, key=lambda c: self._yaw_change(wp, c))
                    path.append(wp)
                if wp.is_junction or self._key(wp) != exit_key:
                    continue
                # Continue along the exit road far enough for every LOOKAHEAD.
                while len(path) * LOOKAHEAD_STEP < 2 * LOOKAHEAD[-1]:
                    nxt = wp.next(LOOKAHEAD_STEP)
                    if not nxt:
                        break
                    wp = min(nxt, key=lambda c: self._yaw_change(wp, c))
                    path.append(wp)
                return path
        return None

    def _lane_counts(self, wp):
        """Same-direction driving lanes to the (left, right) of the ego lane,
        as a lane detector counting markings would report."""
        if wp is None or wp.is_junction:
            return 0, 0
        counts = []
        for step in (lambda w: w.get_left_lane(), lambda w: w.get_right_lane()):
            n, other = 0, step(wp)
            while (other is not None and n < MAX_LANES
                   and other.lane_type == carla.LaneType.Driving
                   and other.lane_id * wp.lane_id > 0):
                n += 1
                other = step(other)
            counts.append(n)
        return counts[0], counts[1]

    @staticmethod
    def _to_vehicle_xy(vehicle_transform, point):
        yaw = math.radians(vehicle_transform.rotation.yaw)
        dx = point.x - vehicle_transform.location.x
        dy = point.y - vehicle_transform.location.y
        return dx * math.cos(yaw) + dy * math.sin(yaw), \
            -dx * math.sin(yaw) + dy * math.cos(yaw)

    def _line_features(self, vehicle_tf, wp, side):
        if wp is None:
            return [0.0, 0.0, 0.0, 0.0]
        if wp.is_junction:
            # No paint inside junctions: give the edges of the route's
            # connector, marked virtual so they differ from real lines.
            code = VIRTUAL_MARKING
        else:
            mark = wp.left_lane_marking if side == "left" else wp.right_lane_marking
            code = self._marking_code(mark)
            if code == 0:
                return [0.0, 0.0, 0.0, 0.0]
        # In CARLA's local frame, positive y points right.
        right = wp.transform.get_right_vector()
        scale = (-0.5 if side == "left" else 0.5) * wp.lane_width
        origin = wp.transform.location
        boundary = carla.Location(
            x=origin.x + scale * right.x,
            y=origin.y + scale * right.y,
            z=origin.z + scale * right.z,
        )
        x, y = self._to_vehicle_xy(vehicle_tf, boundary)
        return [float(np.clip(x, -60.0, 60.0)),
                float(np.clip(y, -60.0, 60.0)), float(code), 1.0]

    def _traffic_control(self, lane_wp):
        """Traffic light or stop sign controlling the ego lane at the next
        junction, as a camera would see it: (kind, distance, id, light actor).

        Town10 places its signals on the connector roads just inside the
        junction, so they are looked up past the junction entry and kept only
        if they belong to that junction. Distance is to the junction entry
        (the stop line).
        """
        if lane_wp is None or lane_wp.is_junction:
            return None
        wp, s, junction_id = lane_wp, 0.0, None
        while s < CONTROL_VIEW_M:
            choices = wp.next(LOOKAHEAD_STEP)
            if not choices:
                return None
            wp = min(choices, key=lambda c: self._yaw_change(wp, c))
            s += LOOKAHEAD_STEP
            if wp.is_junction:
                junction_id = wp.get_junction().id
                break
        if junction_id is None:
            return None
        nearest = None
        for lm in lane_wp.get_landmarks(s + 5.0, False):
            if lm.type not in (LIGHT_TYPE, STOP_TYPE):
                continue
            lm_wp = lm.waypoint
            if lm_wp.is_junction and lm_wp.get_junction().id != junction_id:
                continue
            if nearest is None or lm.distance < nearest.distance:
                nearest = lm
        if nearest is None:
            return None
        distance = s - LOOKAHEAD_STEP  # last non-junction point = stop line
        if nearest.type == STOP_TYPE:
            return ("stop", distance, nearest.id, None)
        if nearest.id not in self._light_cache:
            self._light_cache[nearest.id] = self.world.get_traffic_light(nearest)
        light = self._light_cache[nearest.id]
        if light is None:
            return None
        kind = {carla.TrafficLightState.Red: "red",
                carla.TrafficLightState.Yellow: "yellow",
                carla.TrafficLightState.Green: "green"}.get(light.get_state())
        return None if kind is None else (kind, distance, nearest.id, light)

    def _control_features(self):
        onehot = [0.0] * len(CONTROL_KINDS)
        if self._control is None:
            return onehot + [100.0, 0.0]
        kind, distance, landmark_id, _ = self._control
        onehot[CONTROL_KINDS.index(kind)] = 1.0
        stopped = float(kind == "stop" and self._stopped_sign == landmark_id)
        return onehot + [min(distance, 100.0), stopped]

    def _parked_vehicles(self):
        """Locations of parked cars (map objects, not actors)."""
        labels = [getattr(carla.CityObjectLabel, name) for name in
                  ("Car", "Truck", "Bus", "Motorcycle", "Bicycle", "Vehicles")
                  if hasattr(carla.CityObjectLabel, name)]
        locations = []
        for label in labels:
            for obj in self.world.get_environment_objects(label):
                locations.append(obj.bounding_box.location)
        return locations

    def _remove_old_traffic(self):
        """Destroy traffic left behind by an earlier run that did not close()."""
        for actor in self.world.get_actors():
            if actor.type_id.startswith("controller.ai.walker"):
                actor.stop()
                actor.destroy()
            elif actor.type_id.startswith("walker.pedestrian"):
                actor.destroy()
            elif (actor.type_id.startswith("vehicle.")
                  and actor.attributes.get("role_name") == "npc"):
                actor.destroy()
        self.world.tick()

    def _spawn_traffic(self):
        """Top up autopilot cars and AI pedestrians to the requested counts."""
        hero = self.vehicle.get_location()
        library = self.world.get_blueprint_library()
        self._npcs = [a for a in self._npcs if a.is_alive]
        need = self.traffic_vehicles - len(self._npcs)
        if need > 0:
            cars = [bp for bp in library.filter("vehicle.*")
                    if bp.has_attribute("number_of_wheels")
                    and int(bp.get_attribute("number_of_wheels")) == 4]
            points = [p for p in self.spawn_points
                      if self._distance2d(p.location, hero) > 15.0]
            for i in self.np_random.permutation(len(points)):
                if need <= 0:
                    break
                bp = cars[int(self.np_random.integers(len(cars)))]
                bp.set_attribute("role_name", "npc")
                npc = self.world.try_spawn_actor(bp, points[int(i)])
                if npc is not None:
                    npc.set_autopilot(True, self.tm.get_port())
                    self._npcs.append(npc)
                    need -= 1

        alive = [(w, c) for w, c in zip(self._walkers, self._walker_ctrls)
                 if w.is_alive and c.is_alive]
        self._walkers = [w for w, _ in alive]
        self._walker_ctrls = [c for _, c in alive]
        need = self.pedestrians - len(self._walkers)
        if need > 0:
            people = library.filter("walker.pedestrian.*")
            ctrl_bp = library.find("controller.ai.walker")
            new_ctrls = []
            for _ in range(need * 3):
                if need <= 0:
                    break
                loc = self.world.get_random_location_from_navigation()
                if loc is None or self._distance2d(loc, hero) < 15.0:
                    continue
                bp = people[int(self.np_random.integers(len(people)))]
                if bp.has_attribute("is_invincible"):
                    bp.set_attribute("is_invincible", "false")
                walker = self.world.try_spawn_actor(bp, carla.Transform(loc))
                if walker is None:
                    continue
                ctrl = self.world.spawn_actor(ctrl_bp, carla.Transform(), walker)
                self._walkers.append(walker)
                self._walker_ctrls.append(ctrl)
                new_ctrls.append(ctrl)
                need -= 1
            self.world.tick()  # controllers need a tick before they start
            for ctrl in new_ctrls:
                ctrl.start()
                ctrl.go_to_location(self.world.get_random_location_from_navigation())
                ctrl.set_max_speed(1.4)

    def _front_features(self, vehicle_tf, ahead):
        """Front sensor: nearest vehicle or pedestrian on the path ahead.

        The path is the lookahead lane centers (the ego lane, or the arc
        through a junction). Pedestrians count only when on a driving lane,
        not on the sidewalk. Sets self._front for the waiting rule in step().
        """
        self._front = None
        path = [(0.0, 0.0)] + [
            self._to_vehicle_xy(vehicle_tf, wp.transform.location)
            for _, wp in sorted(ahead.items()) if wp is not None]
        ego_v = self.vehicle.get_velocity()
        objects = [(a, True, False) for a in self._npcs if a.is_alive]
        objects += [(a, True, True) for a in self._walkers if a.is_alive]
        objects += [(loc, False, False) for loc in self._parked]
        best = None
        for obj, moving, walker in objects:
            loc = obj.get_location() if moving else obj
            x, y = self._to_vehicle_xy(vehicle_tf, loc)
            dist = math.hypot(x, y)
            if x <= 0.0 or dist > FRONT_RANGE_M or (best and dist >= best[0]):
                continue
            if self._distance_to_path(path, x, y) > PATH_HALF_WIDTH_M:
                continue
            # People on the sidewalk next to the lane are no obstacle; only
            # those actually on the road (crossing, stepping out) are.
            if walker and self.map.get_waypoint(
                    loc, project_to_road=False, lane_type=carla.LaneType.Driving) is None:
                continue
            v = obj.get_velocity() if moving else carla.Vector3D()
            # Closing speed: how fast the gap shrinks (positive = approaching).
            closing = ((ego_v.x - v.x) * (loc.x - vehicle_tf.location.x)
                       + (ego_v.y - v.y) * (loc.y - vehicle_tf.location.y)) / max(dist, 0.1)
            best = (dist, closing, moving, math.hypot(v.x, v.y))
        self._front = best
        if best is None:
            return [0.0, FRONT_RANGE_M, 0.0]
        return [1.0, best[0], float(np.clip(best[1], -30.0, 30.0))]

    @staticmethod
    def _distance_to_path(path, x, y):
        if len(path) < 2:
            return abs(y)
        best = float("inf")
        for (ax, ay), (bx, by) in zip(path[:-1], path[1:]):
            dx, dy = bx - ax, by - ay
            length2 = dx * dx + dy * dy
            t = 0.0 if length2 < 1e-9 else min(1.0, max(0.0, ((x - ax) * dx + (y - ay) * dy) / length2))
            best = min(best, math.hypot(x - ax - t * dx, y - ay - t * dy))
        return best

    def _lane_offset(self, vehicle_tf, lane_wp):
        """(lateral offset m, heading error deg) from the own lane's center,
        as a lane-keeping camera reports it; (0, 0) in junctions/off-lane."""
        if lane_wp is None or lane_wp.is_junction:
            return 0.0, 0.0
        c = lane_wp.transform.location
        r = lane_wp.transform.get_right_vector()
        offset = ((vehicle_tf.location.x - c.x) * r.x
                  + (vehicle_tf.location.y - c.y) * r.y)
        heading = ((vehicle_tf.rotation.yaw - lane_wp.transform.rotation.yaw + 180.0)
                   % 360.0 - 180.0)
        return float(np.clip(offset, -3.0, 3.0)), float(np.clip(heading, -90.0, 90.0))

    def _lane_cost(self, lane_wp):
        offset, heading = self._lane_offset(self.vehicle.get_transform(), lane_wp)
        return LANE_PENALTY * (abs(offset) + abs(math.radians(heading)))

    def _side_features(self, vehicle_tf, lane_wp):
        """Blind-spot / rear radar: nearest vehicle in the left lane, the right
        lane (ahead or behind) and behind in the own lane, within SIDE_RANGE_M.
        Lanes are matched on the map, so curves don't confuse sides."""
        empty = [0.0, 0.0, 0.0]
        if lane_wp is None or lane_wp.is_junction:
            return empty * 3

        def neighbour(wp):
            if (wp is not None and wp.lane_type == carla.LaneType.Driving
                    and wp.lane_id * lane_wp.lane_id > 0):
                return wp.lane_id
            return None

        lanes = {"left": neighbour(lane_wp.get_left_lane()),
                 "right": neighbour(lane_wp.get_right_lane()),
                 "rear": lane_wp.lane_id}
        yaw = math.radians(vehicle_tf.rotation.yaw)
        fx, fy = math.cos(yaw), math.sin(yaw)
        ego_v = self.vehicle.get_velocity()
        ego_fwd = ego_v.x * fx + ego_v.y * fy
        best = {}
        objects = [(a, True) for a in self._npcs if a.is_alive]
        objects += [(loc, False) for loc in self._parked]
        for obj, moving in objects:
            loc = obj.get_location() if moving else obj
            x, y = self._to_vehicle_xy(vehicle_tf, loc)
            if abs(x) > SIDE_RANGE_M or abs(y) > 8.0:
                continue
            wp = self.map.get_waypoint(loc, project_to_road=False,
                                       lane_type=carla.LaneType.Driving)
            if wp is None or wp.road_id != lane_wp.road_id:
                continue
            for slot, lane_id in lanes.items():
                if lane_id is None or wp.lane_id != lane_id:
                    continue
                if slot == "rear" and x >= 0.0:
                    continue
                if slot not in best or abs(x) < abs(best[slot][0]):
                    v = obj.get_velocity() if moving else carla.Vector3D()
                    best[slot] = (x, (v.x * fx + v.y * fy) - ego_fwd)
        features = []
        for slot in ("left", "right", "rear"):
            if slot in best:
                x, rel = best[slot]
                features += [1.0, float(x), float(np.clip(rel, -30.0, 30.0))]
            else:
                features += empty
        return features

    def _speed_and_limit(self):
        v = self.vehicle.get_velocity()
        speed = 3.6 * math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
        raw_limit = float(self.vehicle.get_speed_limit())
        valid = math.isfinite(raw_limit) and 1.0 <= raw_limit <= 200.0
        return speed, (raw_limit if valid else 0.0), valid

    def _get_state(self, status):
        tf = self.vehicle.get_transform()
        speed, limit, _ = self._speed_and_limit()  # limit is 0 when unknown
        cmd, maneuver_dist, _ = self._next_maneuver()
        line_wp = status["waypoint"]
        ahead = {}
        if line_wp is None:
            # Off the lane: no lines.
            all_lines = [[0.0] * 4 for _ in range(2 * len(LOOKAHEAD))]
        else:
            ahead = self._lane_ahead(line_wp)
            all_lines = []
            for distance in LOOKAHEAD:
                ahead_wp = ahead.get(distance)
                for side in ("left", "right"):
                    all_lines.append(self._line_features(tf, ahead_wp, side))

        lanes_left, lanes_right = self._lane_counts(line_wp)
        lines = []
        for i, (x, y, code, _valid) in enumerate(all_lines):
            # all_lines is ordered (d, side); the first two are at d=0.
            lines += [y, code] if i < 2 else [x, y, code]
        values = (
            [speed, limit, self.prev_steer]
            + cmd + [maneuver_dist]
            + [float(lanes_left), float(lanes_right)]
            + self._control_features()
            + self._front_features(tf, ahead)
            + list(self._lane_offset(tf, status["waypoint"]))
            + self._side_features(tf, status["waypoint"])
            + lines
        )
        obs = np.asarray(values, dtype=np.float32)
        return np.clip(obs, self.observation_space.low,
                       self.observation_space.high).astype(np.float32)

    @staticmethod
    def _is_forbidden_lane_change(before, after):
        if before is None or after is None:
            return False
        if before.is_junction or after.is_junction:
            return False
        if before.road_id != after.road_id or before.lane_id == after.lane_id:
            return False
        if (before.lane_id > 0) != (after.lane_id > 0):
            return False  # opposite-direction lane is handled by wrong_way
        a = before.transform.location
        b = after.transform.location
        right = before.transform.get_right_vector()
        side_dot = (b.x - a.x) * right.x + (b.y - a.y) * right.y
        lane_change = before.lane_change
        permitted = (
            lane_change in (carla.LaneChange.Right, carla.LaneChange.Both)
            if side_dot >= 0 else
            lane_change in (carla.LaneChange.Left, carla.LaneChange.Both)
        )
        return not permitted or abs(after.lane_id - before.lane_id) > 1

    def _consume_forbidden_invasions(self, before, after):
        """Sensor catches solid crossings even when lane center has not switched.

        Mixed SolidBroken/BrokenSolid markings are direction-dependent, so
        `_is_forbidden_lane_change` handles them through OpenDRIVE permissions.
        """
        forbidden = False
        while self._lane_events:
            frame, marking_types = self._lane_events.popleft()
            if frame < self.episode_start_frame:
                continue
            if before is None or after is None:
                continue
            if before.is_junction or after.is_junction:
                continue
            if before.road_id != after.road_id:
                continue
            if "Solid" in marking_types or "SolidSolid" in marking_types:
                forbidden = True
        return forbidden

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        self._spawn_traffic()
        self._occupied = [a.get_location() for a in self._npcs + self._walkers
                          if a.is_alive]
        self._occupied_cars = [a.get_location() for a in self._npcs if a.is_alive]
        # Where roads overlap (junctions), the settled vehicle can map to a
        # different road than its spawn point did; try another route then.
        for _ in range(20):
            self._sample_route()
            self.vehicle.set_simulate_physics(False)
            self.vehicle.set_transform(self.start_transform)
            self.vehicle.set_target_velocity(carla.Vector3D())
            self.vehicle.set_target_angular_velocity(carla.Vector3D())
            self.vehicle.set_simulate_physics(True)
            frame = self.world.tick()
            status = self._check_route_status(self.vehicle.get_location())
            if status["on_route"]:
                break
        else:
            wp = status["waypoint"]
            raise RuntimeError(
                "Reset placed vehicle outside the planned start lane: vehicle "
                f"{None if wp is None else self._key(wp)}, route starts "
                f"{[s['key'] for s in self.route_segments[:3]]}"
            )

        self.steps = 0
        self.stuck_steps = 0
        self.prev_steer = 0.0
        self._maneuver_idx = 0
        self._episode_parts = {}
        # Discard notifications generated by moving/resetting the vehicle.
        self.episode_start_frame = frame + 1
        self.collided = False
        self._lane_events.clear()
        self.prev_lane_wp = status["waypoint"]
        self._latched_control = None
        self._stopped_sign = None
        self._stop_wait = 0
        self._jpath = None
        self._control = self._traffic_control(status["waypoint"])
        obs = self._get_state(status)
        return obs, {
            "route_length_m": self.route_total,
            "start_road_id": self.route[0].road_id,
            "goal_road_id": self.route[-1].road_id,
        }

    def step(self, action):
        steer = float(np.clip(action[0], -1.0, 1.0))
        longitudinal = float(np.clip(action[1], -1.0, 1.0))
        # Zero-mean exploration should still roll forward; only a clearly
        # negative command brakes, with a small coasting band in between.
        throttle = float(np.clip(0.25 + 0.75 * longitudinal, 0.0, 1.0))
        brake = float(np.clip((-longitudinal - 0.6) / 0.4, 0.0, 1.0))

        before_location = self.vehicle.get_location()
        before_wp = self.prev_lane_wp
        self.vehicle.apply_control(carla.VehicleControl(
            steer=steer, throttle=throttle, brake=brake
        ))
        self.world.tick()
        self.steps += 1
        self.prev_steer = steer

        location = self.vehicle.get_location()
        moved_m = self._distance2d(before_location, location)
        status = self._check_route_status(location)
        wp = status["waypoint"]
        progress_m = self._update_progress(status, moved_m)
        speed, speed_limit, speed_limit_valid = self._speed_and_limit()

        # Traffic control seen this step; remember the last one passed so the
        # rule can be checked when the car actually enters the junction.
        control = self._control = self._traffic_control(wp)
        waiting_ok = False
        if control is not None:
            kind, distance, landmark_id, _ = control
            self._latched_control = control
            if kind in ("red", "yellow") and distance <= LIGHT_WAIT_M:
                waiting_ok = True
            elif kind == "stop" and distance <= STOP_ZONE_M + 4.0:
                if (self._stopped_sign != landmark_id and distance <= STOP_ZONE_M
                        and speed < STOP_SPEED_KMH):
                    self._stopped_sign, self._stop_wait = landmark_id, 0
                if self._stopped_sign == landmark_id:
                    self._stop_wait += 1
                waiting_ok = (self._stopped_sign != landmark_id
                              or self._stop_wait <= STOP_WAIT_STEPS)

        entering_junction = (wp is not None and wp.is_junction
                             and before_wp is not None and not before_wp.is_junction)
        control_violation = None
        if entering_junction and self._latched_control is not None:
            kind, _, landmark_id, light = self._latched_control
            if kind == "stop":
                if self._stopped_sign != landmark_id:
                    control_violation = "stop_sign"
            elif light is not None and light.get_state() == carla.TrafficLightState.Red:
                control_violation = "red_light"
            self._latched_control = None
            self._stopped_sign, self._stop_wait = None, 0

        # Waiting behind a car or pedestrian close ahead is fine too, whatever
        # its speed (not a parked car, which has to be driven around).
        front = self._front
        if front is not None and front[2] and front[0] <= QUEUE_GAP_M:
            waiting_ok = True

        # Waiting at a red light or a stop sign is not being stuck.
        if speed < 2.0 and not waiting_ok:
            self.stuck_steps += 1
        else:
            self.stuck_steps = 0
        stuck = self.stuck_steps >= self.stuck_timeout_steps

        off_road = status["off_road"]

        # No bonus for speed itself (it pushed the car into turns too fast):
        # only penalties above the limit and for crawling below MIN_SPEED_KMH.
        speed_reward = 0.0
        if speed_limit_valid:
            ratio = speed / speed_limit
            # A linear overspeed penalty must outweigh the extra progress
            # reward from driving above the legal limit at fixed time steps.
            speed_reward -= 2.0 * min(max(0.0, ratio - 1.0), 5.0)
        if speed < MIN_SPEED_KMH and not waiting_ok:
            # Grows as the car slows: stopped costs about what driving at
            # 15 km/h earns in progress, so creeping is never the safe choice.
            speed_reward -= LOW_SPEED_PENALTY * (MIN_SPEED_KMH - speed) / MIN_SPEED_KMH
        speed_reward *= self.step_scale

        illegal_by_map = self._is_forbidden_lane_change(before_wp, wp)
        illegal_by_sensor = self._consume_forbidden_invasions(before_wp, wp)
        illegal_crossing = illegal_by_map or illegal_by_sensor

        # Dense reward: actual path progress (signed), not lane center/heading.
        parts = {
            "progress": 1.5 * progress_m,
            "speed": speed_reward,
            "step": -0.02 * self.step_scale,
            "illegal_crossing": -5.0 if illegal_crossing else 0.0,
            "lane": -self._lane_cost(wp),
        }

        distance_to_goal = self._distance2d(location, self.goal_location)
        reached_goal = (
            status["on_route"] and distance_to_goal <= 5.0
            and self.route_total - self.route_s <= 12.0
        )
        if self.collided:
            reason = "collision"
        elif off_road:
            reason = "off_road"
        elif status["wrong_road"]:
            reason = "wrong_road"
        elif status["wrong_way"]:
            # Oncoming lanes end the episode like leaving the road: crossing
            # into them is a crash-level mistake, not a per-step cost.
            reason = "wrong_way"
        elif control_violation:
            reason = control_violation
        elif stuck:
            reason = "stuck"
        elif reached_goal:
            reason = "goal"
        else:
            reason = None
        terminated = reason is not None
        truncated = bool(self.steps >= self.max_steps and not terminated)
        if truncated:
            reason = "time_limit"
        if reason in ("collision", "off_road", "wrong_road", "wrong_way", "red_light", "stop_sign", "stuck"):
            # Stuck must cost as much as a crash, or stopping before a hard
            # junction becomes the cheapest way to end the episode.
            parts = {"terminal": -30.0}
        elif reason == "goal":
            parts["terminal"] = 50.0
        reward = sum(parts.values())
        for name, value in parts.items():
            self._episode_parts[name] = self._episode_parts.get(name, 0.0) + value
        if reason in ("collision", "off_road", "wrong_road", "wrong_way",
                      "red_light", "stop_sign"):
            self._print_failure(reason, location, wp, speed)

        # Junction outcome per maneuver, for the training log only (no reward).
        # Passed = progress went beyond the maneuver's last route point;
        # failed = the episode failed at or within JUNCTION_ATTEMPT_M of it.
        maneuver_results = []
        while (self._maneuver_idx < len(self.turn_events)
               and self.route_s_values[self.turn_events[self._maneuver_idx][1]] < self.route_s):
            maneuver_results.append((self.turn_events[self._maneuver_idx][2].name, True))
            self._maneuver_idx += 1
        if (reason in ("collision", "off_road", "wrong_road", "wrong_way", "red_light", "stop_sign", "stuck")
                and self._maneuver_idx < len(self.turn_events)):
            start, _, option = self.turn_events[self._maneuver_idx]
            if self.route_s >= self.route_s_values[start] - JUNCTION_ATTEMPT_M:
                maneuver_results.append((option.name, False))

        obs = self._get_state(status)
        self.prev_lane_wp = wp
        info = {
            "termination_reason": reason,
            "maneuver_results": maneuver_results,
            # Per-episode sum of each reward term, only on the last step.
            "reward_parts": dict(self._episode_parts) if reason else None,
            "route_progress_m": self.route_s,
            "route_remaining_m": max(0.0, self.route_total - self.route_s),
        }
        return obs, float(reward), bool(terminated), bool(truncated), info

    def _print_failure(self, reason, location, wp, speed):
        """One diagnostic line per failed episode (goes to train_v15.log)."""
        d = np.hypot(self.route_xy[:, 0] - location.x, self.route_xy[:, 1] - location.y)
        nearest = int(np.argmin(d))
        upcoming = [s["key"] for s in
                    self.route_segments[self.route_segment_idx:self.route_segment_idx + 3]]
        cmd, maneuver_dist, maneuver_valid = self._next_maneuver()
        maneuver = TURN_OPTIONS[cmd.index(1.0)].name if maneuver_valid else "NONE"
        print(
            f"[fail] {reason} step={self.steps} speed={speed:.1f}km/h "
            f"vehicle={None if wp is None else self._key(wp)} "
            f"junction={None if wp is None else wp.is_junction} "
            f"expected={upcoming} route_dist={d[nearest]:.2f}m "
            f"nearest_route_key={self.route_keys[nearest]} "
            f"progress={self.route_s:.0f}/{self.route_total:.0f}m maneuver={maneuver} "
            f"maneuver_dist={maneuver_dist:.0f}m",
            flush=True,
        )

    def close(self):
        for sensor in self._sensors:
            if sensor is not None and sensor.is_alive:
                sensor.stop()
                sensor.destroy()
        self._sensors.clear()
        if self.vehicle is not None and self.vehicle.is_alive:
            self.vehicle.apply_control(carla.VehicleControl(
                throttle=0.0, brake=1.0
            ))
        # Leave synchronous mode first: in it the server only handles destroy
        # requests while someone ticks, so destroying first just times out.
        if self.tm is not None:
            self.tm.set_synchronous_mode(False)
        self.world.apply_settings(self.original_settings)
        for ctrl in self._walker_ctrls:
            if ctrl.is_alive:
                ctrl.stop()
        actors = [a for a in self._walker_ctrls + self._walkers + self._npcs if a.is_alive]
        if actors:
            self.client.apply_batch([carla.command.DestroyActor(a) for a in actors])
