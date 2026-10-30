HERO_ROLE = "hero"


def find_hero(world):
    """Return the vehicle spawned by spawn_vehicle.py (role_name "hero").

    Falls back to the only vehicle in the world, so a car spawned before
    role names were used still works.
    """
    vehicles = world.get_actors().filter("vehicle.*")
    for vehicle in vehicles:
        if vehicle.attributes.get("role_name") == HERO_ROLE:
            return vehicle
    if len(vehicles) == 1:
        return vehicles[0]
    raise RuntimeError("Hero vehicle not found (run spawn_vehicle.py)")
