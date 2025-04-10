from math import atan, degrees
import math
from operator import attrgetter
from pprint import pprint
from pytz import timezone
import datetime as dt
from skyfield.api import load, wgs84, Star, Topos
from skyfield.data import hipparcos
from skyfield.framelib import itrs
from skyfield.positionlib import Apparent
from skyfield import almanac
from skyfield.named_stars import named_star_dict
from skyfield.units import Angle
import numpy as np
import pandas as pd
from astroquery.simbad import Simbad
from astropy.table import Table

# do some setup of common things, like load the catalog, earth

with load.open(hipparcos.URL) as f:
            df = hipparcos.load_dataframe(f)

# # Create a MultiIndex with the desired levels
# new_index = pd.MultiIndex.from_arrays(
#     [
#         df['ra_hours'],
#         df['dec_degrees'],
#         df['magnitude'],
#         df.index,  # HIP ID
#     ],
#     names=['ra_hours', 'dec_degrees', 'magnitude', 'hip']
# )

# # Reindex the dataframe
# reindexed_df = df.set_index(new_index)

# # Sort the index for efficient lookup
# reindexed_df = reindexed_df.sort_index()

# df = reindexed_df

planets = load('de421.bsp')
earth = planets['earth']

def get_hd_number_from_simbad(hip_number):
    """
    Looks up the HD catalog number of a star in SIMBAD given its HIP number.

    Args:
        hip_number (int): The Hipparcos (HIP) number of the star.

    Returns:
        str or None: The HD catalog number (e.g., "HD 12345") if found, otherwise None.
    """
    try:
        # Configure Simbad to return HD numbers
        Simbad.add_votable_fields('ids')

        # Query Simbad for the star with the given HIP number
        result_table = Simbad.query_object(f'HIP{hip_number}')

        if result_table is None or len(result_table) == 0:
            print(f"No SIMBAD entry found for HIP {hip_number}")
            return None

        # Extract HD numbers from the result table
        hd_numbers = []
        for row in result_table:
            if 'HD' in row['main_id']:
                hd_numbers.append(row['main_id'])

        if not hd_numbers:
            print(f"No HD number found for HIP {hip_number}")
            return None

        # Return the first HD number found (there might be multiple)
        return hd_numbers[0]

    except Exception as e:
        print(f"An error occurred while querying SIMBAD for HIP {hip_number}: {e}")
        return None

def find_stars_at_altaz(df, latitude, longitude, time, target_azimuth, target_altitude, tolerance_degrees=1.0, magnitude=4.0):
    """
    Finds stars within a given tolerance of a target azimuth and altitude.

    Args:
        df: the HIP dataframe
        latitude (float): Latitude of the observer (degrees).
        longitude (float): Longitude of the observer (degrees).
        time (skyfield.timelib.Time): The time of observation.
        target_azimuth (float): Target azimuth (degrees, 0=North, increasing clockwise).
        target_altitude (float): Target altitude (degrees, 0=horizon, 90=zenith).
        tolerance_degrees (float): Tolerance in degrees for matching azimuth and altitude.
        magnitude (float): Filter stars with a magnitude less than this value (big speedup)

    Returns:
        list: A list of dictionaries, each containing star information (name, HIP number, azimuth, altitude).
    """

    # Load necessary data
    ts = load.timescale()
    
    # filter on magnitude
    df = df.loc[(df['magnitude'] <= magnitude)]

    
    # Create the observer
    observer_location = wgs84.latlon(latitude, longitude)
    observer = earth + observer_location

    # Calculate the apparent place of the stars
    astrometric = observer.at(time)

    # Convert target azimuth and altitude to radians
    target_azimuth_rad = np.deg2rad(target_azimuth)
    target_altitude_rad = np.deg2rad(target_altitude)

    # Create an empty list to store the results
    found_stars = []

    # Iterate through the stars in the catalog
    for hip_number, row in df.iterrows():
        star = Star.from_dataframe(row)
        apparent = astrometric.observe(star).apparent()
        alt, az, _ = apparent.altaz()



        # Check if the star is within the tolerance
        if (abs(az.radians - target_azimuth_rad) <= np.deg2rad(tolerance_degrees) and
                abs(alt.radians - target_altitude_rad) <= np.deg2rad(tolerance_degrees)):
            found_stars.append({
                'hip': hip_number,
                'azimuth': np.rad2deg(az.radians),
                'altitude': np.rad2deg(alt.radians),
                'magnitude': row['magnitude'],
                'ra': row['ra_hours'],
                'dec': row['dec_degrees']
            })

    return found_stars

def find_stars_at_ra_dec(df, latitude, longitude, time, target_ra, target_dec, tolerance_degrees, tolerance_hours, magnitude):
    """
    Finds stars within a given tolerance of a target azimuth and altitude.

    Args:
        df: the HIP dataframe
        latitude (float): Latitude of the observer (degrees).
        longitude (float): Longitude of the observer (degrees).
        time (skyfield.timelib.Time): The time of observation.
        target_ra (float): Target right ascention (hours)
        target_dec (float): Target declination (degrees)
        tolerance_degrees (float): Tolerance in degrees for matching azimuth and altitude.
        tolerance_hours: Tolerance in hours for matching RA and Dec
        magnitude (float): Filter stars with a magnitude less than this value (big speedup)

    Returns:
        list: A list of dictionaries, each containing star information (name, HIP number, azimuth, altitude).
    """
    print(f"Searching for stars within {tolerance_hours:2f} hours of RA {target_ra:2f} and {tolerance_degrees:2f} degrees of Dec {target_dec:2f}, magnitude {magnitude}")
    
    # filter on magnitude
    df = df.loc[(df['magnitude'] <= magnitude)]
    # Create the observer
    observer_location = wgs84.latlon(latitude, longitude)
    observer = earth + observer_location

    # Calculate the apparent place of the stars
    astrometric = observer.at(time)

    # Create an empty list to store the results
    found_stars = []

    # Iterate through the stars in the catalog
    # returns the index and row (hip_number is index value)
    for hip_number, row in df.iterrows():
        star = Star.from_dataframe(row)
        apparent = astrometric.observe(star).apparent()
        ra, dec, _ = apparent.radec()
        alt,az, _ = apparent.altaz()


        # TODO: We can mass process the entire catalog - might be faster
        # then iterate teh results
        # Check if the star is within the tolerance
        if (abs(ra.hours - target_ra) <= tolerance_hours and
                abs(dec.degrees - target_dec) <= tolerance_degrees):
            
            # lookup star hip number in SIMBAD
            hd_number = get_hd_number_from_simbad(hip_number)    

            found_stars.append({
                'hip': hip_number,
                'hd': hd_number,
                'azimuth': az,
                'altitude': alt,
                'magnitude': row['magnitude'],
                'ra': row['ra_hours'],
                'dec': row['dec_degrees']
            })

    return found_stars

def altaz_to_radec(altitude_degrees, azimuth_degrees, latitude_degrees, longitude_degrees, elevation_meters, timestamp):
    """
    Converts altitude/azimuth coordinates to RA/Dec using Skyfield.

    Args:
        altitude_degrees: Altitude in degrees.
        azimuth_degrees: Azimuth in degrees (clockwise from North).
        latitude_degrees: Latitude of the observer in degrees.
        longitude_degrees: Longitude of the observer in degrees.
        elevation_meters: Elevation of the observer above sea level in meters.
        timestamp: A Skyfield Time object representing the observation time.

    Returns:
        A tuple containing:
            - ra_hours: Right Ascension in hours.
            - dec_degrees: Declination in degrees.
    """

    geographic = wgs84.latlon(latitude_degrees, longitude_degrees, elevation_meters)
    observer = geographic.at(timestamp)
    pos = observer.from_altaz(alt_degrees=altitude_degrees, az_degrees=azimuth_degrees)

    ra, dec, distance = pos.radec()

    return ra.hours, dec.degrees

def calculate_distance_and_altaz(lat1, lng1, elevation1, lat2, lng2, elevation2, timestamp):
    """
    Calculates the distance and alt/az between two locations using Skyfield.

    Args:
        lat1: Latitude of the first location in degrees.
        lng1: Longitude of the first location in degrees.
        elevation1: Elevation of the first location in meters.
        lat2: Latitude of the second location in degrees.
        lng2: Longitude of the second location in degrees.
        elevation2: Elevation of the second location in meters.
        timestamp: A Skyfield Time object representing the observation time.

    Returns:
        A tuple containing:
            - distance_km: The distance between the locations in kilometers.
            - azimuth_degrees: The azimuth from location 1 to location 2 in degrees.
            - altitude_degrees: The altitude from location 1 to location 2 in degrees.
    """
    

    location1 = earth + wgs84.latlon(lat1, lng1, elevation1)
    location2 = earth + wgs84.latlon(lat2, lng2, elevation2)

    # Calculate the difference vector between the two locations at the given time
    difference = location2 - location1

    # Calculate the distance between the locations
    distance = difference.at(timestamp).distance()
    distance_km = distance.km

    # Calculate the alt/az of location 2 as seen from location 1
    location = location1.at(timestamp)
    obs = location.observe(location2)
    altaz = obs.apparent().altaz()

    altitude_degrees = altaz[0].degrees
    azimuth_degrees = altaz[1].degrees

    return distance_km, azimuth_degrees, altitude_degrees

def haversine_distance(lat1_deg, lon1_deg, lat2_deg, lon2_deg, earth_radius_km=6371.0):
    """
    Calculate the great-circle distance between two points
    on the earth (specified in decimal degrees) using the Haversine formula.

    Args:
        lat1_deg: Latitude of point 1 in degrees.
        lon1_deg: Longitude of point 1 in degrees.
        lat2_deg: Latitude of point 2 in degrees.
        lon2_deg: Longitude of point 2 in degrees.
        earth_radius_km: Mean radius of Earth in kilometers. Defaults to 6371.0 km.

    Returns:
        Distance in kilometers.
    """
    # Convert latitude and longitude from degrees to radians
    lat1_rad = math.radians(lat1_deg)
    lon1_rad = math.radians(lon1_deg)
    lat2_rad = math.radians(lat2_deg)
    lon2_rad = math.radians(lon2_deg)

    # Haversine formula
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)) # Central angle in radians

    distance = earth_radius_km * c
    return distance

def time_format(zone, observation_time):
    return observation_time.astimezone(zone).strftime("%d %B %Y %I:%M%p")

if __name__ == "__main__":

    ts = load.timescale()
    zone = timezone('US/Mountain')
    observation_time = dt.datetime(2025, 1, 30, 18, 29, 00, tzinfo=zone)

    time = ts.from_datetime(observation_time)
    # for a given alt/az, all the stars that pass through will be a fixed declination, with ra varying with time
    # set tight dec tolerance, low RA
    tolerance_dec = 10/60.0
    tolerance_ra = 20/60.0
    magnitude = 10.0

    # te 39° 0'0.34"Nst our azimuth calc
    # Location 1: Heffron 38.97605656360721, -104.47841397017535 38°58'13.86" N 104°29'36.05" W
    #  39°00'04.93" N 104°30'10.23" W
    lat1 = Angle(degrees = 39.0 + 0/60.0 + 4.93/3600.0 )  # Degrees
    lng1 = Angle(degrees = -1.0*(104.0 + 30/60.0 + 10.23/3600.0)) # Degrees
    elevation1 = 2073  # Meters


    # Location 2: Pikes peak 38.8409° N, 105.0423° W
    lat2 = Angle(degrees = 38.8409)
    lng2 = Angle(degrees = -105.0423) 
    elevation2 = 4345.0

    # Calculate distance and alt/az
    distance_km, azimuth_degrees, altitude_degrees = calculate_distance_and_altaz(
        lat1.degrees, lng1.degrees, elevation1, lat2.degrees, lng2.degrees, elevation2, time
    )

    distance_flat, azimuth_flat, altitude_flat = calculate_distance_and_altaz(
        lat1.degrees, lng1.degrees, 0, lat2.degrees, lng2.degrees, 0, time
    )

    target_ra, target_dec = altaz_to_radec(altitude_degrees, azimuth_degrees,
                                            lat2.degrees, lng2.degrees, elevation2, time)
    
    flat_angle = degrees(atan((elevation2 - elevation1) / (distance_flat * 1000.0)))

    # calculate the surface distance, using haversine
    distance_arc = haversine_distance(lat1.degrees, lng1.degrees, lat2.degrees, lng2.degrees)

        # 2. Create Skyfield geographic position objects
    location1 = wgs84.latlon(latitude_degrees=lat1.degrees, longitude_degrees=lng1.degrees, elevation_m=0)
    location2 = wgs84.latlon(latitude_degrees=lat2.degrees, longitude_degrees=lng2.degrees, elevation_m=0)

    # 4. Get the geocentric position vectors *at that time*
    #    These are instances of position objects like Geometric, Astrometric etc.
    pos1 = location1.at(time)
    pos2 = location2.at(time)

    # 5. Calculate the angular separation *between the position vectors*
    angle = pos1.separation_from(pos2) # This works!

    # 6. Choose Earth radius (e.g., WGS84 equatorial radius from Skyfield)
    earth_radius_km = wgs84.radius.km
    # earth_radius_km = 6371.0 # Or mean radius

    # 7. Calculate the great-circle distance
    distance_km_skyfield = angle.radians * earth_radius_km
        
    

    print(f"At {time.utc_strftime('%Y-%m-%d %H:%M:%S UTC')}:")
    print(f"  Distance between locations los: {distance_km:.2f} km, On base: {distance_flat:.2f} km, hversine: {distance_arc:.2f} km, Skyfiled: {distance_km_skyfield:.2f} km")
    print(f"  Azimuth from location 1 to location 2: {azimuth_degrees:.2f}°")
    print(f"  Altitude from location 1 to location 2: {altitude_degrees:.2f}° Flat Angle: {flat_angle:.2f}°")
    print(f"  Target RA of alt/az: {target_ra}, Target Dec; {target_dec}")
    print(f"  Target lat/long: {lat2.degrees}, {lng2.degrees} at {elevation2}")
    print()

    exit()
    # found_stars = find_stars_at_altaz(df, latitude, longitude, time, target_azimuth, target_altitude, tolerance_degrees, magnitude)
    found_stars = find_stars_at_ra_dec(df, lat2.degrees, lng2.degrees, time,
                                        target_ra, target_dec, tolerance_dec, tolerance_ra, magnitude)

    if found_stars:
        hip_to_name = {v: k for k, v in named_star_dict.items()}

       


        print(f"Stars found within {tolerance_dec:2f} degrees of Azimuth {azimuth_degrees} and Altitude {altitude_degrees} at {time_format(zone, observation_time)}.")

        for star in sorted(found_stars, key=lambda item: item['magnitude'] ):
             
            print(f" (HIP {star['hip']}): ({star['hd']}) Azimuth = {star['azimuth'].degrees:2f}°, Altitude = {star['altitude'].degrees:.2f}°, Magnitude = {star['magnitude']} RA: {star['ra']:2f} Dec: {star['dec']:2f} {hip_to_name.get(star['hip'], "-")}")
    else:
        print(f"No stars found within {tolerance_dec:2f} degrees of Azimuth {azimuth_degrees:2f} and Altitude {altitude_degrees:2f} at {time_format(zone, observation_time)}.")
