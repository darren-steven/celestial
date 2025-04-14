import numpy as np
from skyfield.api import load, N, W, E, S, PlanetaryConstants # For units if needed
from skyfield import almanac
from skyfield.units import Angle
from scipy.optimize import minimize_scalar

import datetime


# --- Configuration ---
# Setup loader (using a specific directory is good practice)
# On Linux/macOS: '~/skyfield-data'
# On Windows: r'C:\Users\YourUser\skyfield-data' (replace YourUser)
# Or simply use the default temporary location if you omit the path

# Load ephemeris and timescale
ts = load.timescale()
# Using a more recent ephemeris like de440.bsp is generally recommended
try:
    eph = load('de440.bsp')
except Exception:
    print("Could not load de440.bsp, falling back to de421.bsp")
    eph = load('de421.bsp')

SUN_RADIUS_KM_HARDCODED = 695700.0
MOON_RADIUS_KM_HARDCODED = 1737.4
EARTH_EQUATORIAL_RADIUS_KM = 6378.137 # WGS84 value

# Define bodies from the ephemeris
sun = eph['sun']
moon = eph['moon']
earth = eph['earth']

# Define time range for search
t_start = ts.utc(2021, 1, 1)
t_end = ts.utc(2030, 12, 31)


# --- Helper Functions ---

def get_sun_moon_separation_and_radii(t):
    """
    Calculates geocentric apparent angular separation and angular radii
    of Sun and Moon at a given Skyfield Time object 't'.
    Assumes a PCK file with radii has been loaded.

    Returns:
        tuple: (separation_angle, sun_radius_angle, moon_radius_angle)
               Angles are Skyfield Angle objects.
               Returns (None, None, None) if calculation fails or radii are missing.
    """
    try:
        e_at_t = earth.at(t)
        sun_app = e_at_t.observe(sun).apparent()
        moon_app = e_at_t.observe(moon).apparent()

        separation = sun_app.separation_from(moon_app)

        sun_dist_km = sun_app.distance().km
        moon_dist_km = moon_app.distance().km
        # if sun_dist_km <= 0 or moon_dist_km <= 0:
        #      return None, None, None


        sun_radius_rad = np.arctan2(SUN_RADIUS_KM_HARDCODED, sun_dist_km)
        moon_radius_rad = np.arctan2(MOON_RADIUS_KM_HARDCODED, moon_dist_km)

        sun_radius_angle = Angle(radians=sun_radius_rad)
        moon_radius_angle = Angle(radians=moon_radius_rad)
        # Earth angular radius seen from Moon (related to parallax)
        earth_radius_at_moon_rad = np.arcsin(EARTH_EQUATORIAL_RADIUS_KM / moon_dist_km)
        earth_radius_at_moon_angle = Angle(radians=earth_radius_at_moon_rad)
        
        return separation, sun_radius_angle, moon_radius_angle, earth_radius_at_moon_angle

    except Exception as e:
        print(f"Warning: Could not calculate positions/radii at {t.utc_iso()}: {e}")
        return None, None, None


# --- Function for find_discrete: Is eclipse possible? ---
# (Code remains the same, relies on corrected helper function)
def eclipse_possible(t):
    separation, sun_radius, moon_radius, earth_at_moon_radians = get_sun_moon_separation_and_radii(t)
    if separation is None or sun_radius is None or moon_radius is None: # Check radii too
        return False
    return separation.radians < (sun_radius.radians + moon_radius.radians + earth_at_moon_radians.radians)

eclipse_possible.step_days = 0.01

# --- Function for minimize_scalar: Separation in degrees ---
# (Code remains the same, relies on corrected helper function)
def separation_degrees(t_tdb_jd):
    t = ts.tdb(jd=t_tdb_jd)
    separation, _, _, _= get_sun_moon_separation_and_radii(t)
    if separation is None:
        return 180.0
    return separation.degrees

print(f"Searching for potential solar eclipses between {t_start.utc_iso()} and {t_end.utc_iso()}...")
try:
    # Step 1: Find time windows
    times, events = almanac.find_discrete(t_start, t_end, eclipse_possible, num=100) # Adjust num if needed

    print(f"Found {len(times)} transition times using find_discrete. ")
    for tm, event in zip(times, events):
        print(tm.utc_iso(), event, get_sun_moon_separation_and_radii(tm))

    # Step 2: Pair transitions to get potential eclipse windows
    eclipse_windows = []
    last_event = eclipse_possible(t_start)
    t_window_start = None
    if last_event == 1:
        t_window_start = t_start

    for i in range(len(events)):
        current_event = events[i]
        if last_event == 0 and current_event == 1:
            t_window_start = times[i]
        elif last_event == 1 and current_event == 0:
            if t_window_start is not None:
                 t_window_end = times[i]
                 eclipse_windows.append((t_window_start, t_window_end))
                 t_window_start = None
            else:
                 eclipse_windows.append((t_start, times[i]))
        last_event = current_event

    if t_window_start is not None:
        eclipse_windows.append((t_window_start, t_end))


    print(f"Identified {len(eclipse_windows)} potential eclipse windows.")

    # Step 3 & 4: Find minimum separation within each window and verify
    found_eclipses = []
    for i, (t_start_window, t_end_window) in enumerate(eclipse_windows):
        if t_start_window.tdb >= t_end_window.tdb:
            print(f"\nSkipping invalid window {i+1}: Start >= End")
            continue

        print(f"\nAnalyzing window {i+1}: {t_start_window.utc_iso()} to {t_end_window.utc_iso()}")

        result = minimize_scalar(
            separation_degrees,
            bounds=(t_start_window.tdb, t_end_window.tdb),
            method='bounded'
        )

        if result.success:
            t_max_eclipse_jd = result.x
            min_sep_deg = result.fun
            t_max_eclipse = ts.tdb(jd=t_max_eclipse_jd)

            # Add small buffer for boundary check
            time_buffer = 1e-6 # ~0.1 seconds
            if abs(t_max_eclipse.tdb - t_start_window.tdb) < time_buffer or abs(t_max_eclipse.tdb - t_end_window.tdb) < time_buffer:
                 print(f"  Minimum found very close to window boundary ({t_max_eclipse.utc_iso()}), potentially unreliable. Skipping.")
                 continue

            print(f"  Minimum separation found: {min_sep_deg:.5f} deg at {t_max_eclipse.utc_iso()}")

            final_sep, final_sun_r, final_moon_r, earth_size_radians = get_sun_moon_separation_and_radii(t_max_eclipse)

            if final_sep is None or final_sun_r is None or final_moon_r is None:
                print("  Verification failed (calculation or radius error).")
                continue

            sum_radii = final_sun_r.radians + final_moon_r.radians + earth_size_radians.radians
            # Check with a tiny tolerance for floating point math
            is_eclipse = final_sep.radians < sum_radii 
            
            print(f"  At minimum: Sep={final_sep.degrees:.5f}°, Sun Radius={final_sun_r.degrees:.5f}°, Moon Radius={final_moon_r.degrees:.5f}°, Sum={sum_radii:.5f}°")

            if is_eclipse:
                print(f"  >>> Confirmed Geocentric Solar Eclipse <<<")
                is_duplicate = False
                for existing in found_eclipses:
                     # Check if times are within ~2 hours (0.1 days)
                     if abs(existing['time'].tdb - t_max_eclipse.tdb) < 0.08:
                         is_duplicate = True
                         print("  (Skipping as likely duplicate of previously found event)")
                         break
                if not is_duplicate:
                     found_eclipses.append({
                         'time': t_max_eclipse,
                         'min_separation': final_sep,
                         'sun_radius': final_sun_r,
                         'moon_radius': final_moon_r
                     })
            else:
                 print("  Minimum separation is not less than sum of radii. (Near miss or grazing)")

        else:
            print(f"  Optimization failed for this window: {result.message}")

    # --- Report Results ---
    print("\n--- Summary of Geocentric Solar Eclipses Found ---")
    if not found_eclipses:
        print("No geocentric solar eclipses found in the specified period.")
    else:
        found_eclipses.sort(key=lambda e: e['time'].tdb)
        for eclipse in found_eclipses:
            t = eclipse['time']
            min_sep = eclipse['min_separation']
            sun_r = eclipse['sun_radius']
            moon_r = eclipse['moon_radius']

            # Basic Classification (Geocentric View) - Refined
            eclipse_type = "Partial"
            if min_sep.radians < abs(sun_r.radians - moon_r.radians):
                if moon_r.radians >= sun_r.radians:
                    eclipse_type = "Total"
                else:
                    eclipse_type = "Annular"
            elif min_sep.radians < min(sun_r.radians, moon_r.radians) * (180/np.pi) : # Check if center passes well within smaller radius disk
                # This condition needs refinement based on exact definition,
                # but aims to capture clear Total/Annular vs edge cases
                if moon_r.radians >= sun_r.radians:
                    eclipse_type = "Total (likely)"
                else:
                    eclipse_type = "Annular (likely)"
            elif min_sep.radians < max(sun_r.radians, moon_r.radians) * (180/np.pi):
                 eclipse_type = "Partial/Edge Annular/Edge Total"


            print(f"\nMaximum Eclipse: {t.utc_strftime('%Y-%m-%d %H:%M:%S')} UTC")
            print(f"  Min Separation: {min_sep.degrees:.5f}° ({min_sep.arcminutes():.2f}')")
            print(f"  Sun Radius:     {sun_r.degrees:.5f}° ({sun_r.arcminutes():.2f}')")
            print(f"  Moon Radius:    {moon_r.degrees:.5f}° ({moon_r.arcminutes():.2f}')")
            print(f"  Geocentric Type Suggestion: {eclipse_type}")

    print("\nNote: Times are for maximum geocentric eclipse. Visibility depends on location.")

except Exception as e:
    print(f"\nAn error occurred during the search: {e}")
    import traceback
    traceback.print_exc()