"""
Build City Database Script
Generates a database of real US cities with attributes for the Find Your Spot app.

This script uses ONLY real US city names and approximate realistic data.
In production, this data would come from public sources like Census Bureau, NOAA, etc.

Data sources that could be used for real data:
- US Census Bureau: Population, demographics, housing
- NOAA: Climate data (temperature, precipitation, sunny days)
- BLS: Cost of living, unemployment
- FBI UCR: Crime statistics
- Tax Foundation: State tax rates
- FAA: Airport data
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Seed for reproducibility
np.random.seed(42)

# Base directory
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Real US Cities with actual coordinates and approximate populations
# Sources: US Census Bureau, Wikipedia
REAL_CITIES = [
    # Format: (name, state, lat, lon, population, region, climate_zone)

    # ===== NORTHEAST =====
    # New York
    ("New York", "NY", 40.7128, -74.0060, 8336817, "Northeast", "humid_continental"),
    ("Buffalo", "NY", 42.8864, -78.8784, 278349, "Northeast", "humid_continental"),
    ("Rochester", "NY", 43.1566, -77.6088, 211328, "Northeast", "humid_continental"),
    ("Syracuse", "NY", 43.0481, -76.1474, 148620, "Northeast", "humid_continental"),
    ("Albany", "NY", 42.6526, -73.7562, 99224, "Northeast", "humid_continental"),
    ("Yonkers", "NY", 40.9312, -73.8987, 211569, "Northeast", "humid_continental"),
    ("Ithaca", "NY", 42.4440, -76.5019, 32108, "Northeast", "humid_continental"),

    # Massachusetts
    ("Boston", "MA", 42.3601, -71.0589, 675647, "Northeast", "humid_continental"),
    ("Worcester", "MA", 42.2626, -71.8023, 206518, "Northeast", "humid_continental"),
    ("Springfield", "MA", 42.1015, -72.5898, 155929, "Northeast", "humid_continental"),
    ("Cambridge", "MA", 42.3736, -71.1097, 118403, "Northeast", "humid_continental"),
    ("Lowell", "MA", 42.6334, -71.3162, 115554, "Northeast", "humid_continental"),

    # Pennsylvania
    ("Philadelphia", "PA", 39.9526, -75.1652, 1584064, "Northeast", "humid_subtropical"),
    ("Pittsburgh", "PA", 40.4406, -79.9959, 302971, "Northeast", "humid_continental"),
    ("Allentown", "PA", 40.6084, -75.4902, 126092, "Northeast", "humid_continental"),
    ("Erie", "PA", 42.1292, -80.0851, 94831, "Northeast", "humid_continental"),
    ("Reading", "PA", 40.3356, -75.9269, 95112, "Northeast", "humid_continental"),
    ("Scranton", "PA", 41.4090, -75.6624, 76997, "Northeast", "humid_continental"),
    ("Bethlehem", "PA", 40.6259, -75.3705, 75781, "Northeast", "humid_continental"),
    ("Lancaster", "PA", 40.0379, -76.3055, 63490, "Northeast", "humid_continental"),
    ("State College", "PA", 40.7934, -77.8600, 42074, "Northeast", "humid_continental"),

    # New Jersey
    ("Newark", "NJ", 40.7357, -74.1724, 311549, "Northeast", "humid_subtropical"),
    ("Jersey City", "NJ", 40.7178, -74.0431, 292449, "Northeast", "humid_subtropical"),
    ("Paterson", "NJ", 40.9168, -74.1718, 159732, "Northeast", "humid_subtropical"),
    ("Trenton", "NJ", 40.2171, -74.7429, 90871, "Northeast", "humid_subtropical"),
    ("Princeton", "NJ", 40.3573, -74.6672, 31822, "Northeast", "humid_subtropical"),

    # Connecticut
    ("Hartford", "CT", 41.7658, -72.6734, 121054, "Northeast", "humid_continental"),
    ("New Haven", "CT", 41.3083, -72.9279, 134023, "Northeast", "humid_continental"),
    ("Stamford", "CT", 41.0534, -73.5387, 135470, "Northeast", "humid_continental"),
    ("Bridgeport", "CT", 41.1865, -73.1952, 148654, "Northeast", "humid_continental"),

    # Rhode Island
    ("Providence", "RI", 41.8240, -71.4128, 190934, "Northeast", "humid_continental"),

    # Vermont
    ("Burlington", "VT", 44.4759, -73.2121, 45489, "Northeast", "humid_continental"),
    ("Montpelier", "VT", 44.2601, -72.5754, 8074, "Northeast", "humid_continental"),

    # New Hampshire
    ("Manchester", "NH", 42.9956, -71.4548, 115644, "Northeast", "humid_continental"),
    ("Concord", "NH", 43.2081, -71.5376, 43976, "Northeast", "humid_continental"),
    ("Portsmouth", "NH", 43.0718, -70.7626, 22158, "Northeast", "humid_continental"),

    # Maine
    ("Portland", "ME", 43.6591, -70.2568, 68408, "Northeast", "humid_continental"),
    ("Bangor", "ME", 44.8016, -68.7712, 31903, "Northeast", "humid_continental"),

    # ===== SOUTHEAST =====
    # Florida
    ("Miami", "FL", 25.7617, -80.1918, 467963, "Southeast", "tropical"),
    ("Tampa", "FL", 27.9506, -82.4572, 399700, "Southeast", "humid_subtropical"),
    ("Orlando", "FL", 28.5383, -81.3792, 307573, "Southeast", "humid_subtropical"),
    ("Jacksonville", "FL", 30.3322, -81.6557, 949611, "Southeast", "humid_subtropical"),
    ("St. Petersburg", "FL", 27.7676, -82.6403, 265351, "Southeast", "humid_subtropical"),
    ("Fort Lauderdale", "FL", 26.1224, -80.1373, 182760, "Southeast", "tropical"),
    ("Tallahassee", "FL", 30.4383, -84.2807, 196169, "Southeast", "humid_subtropical"),
    ("Gainesville", "FL", 29.6516, -82.3248, 141085, "Southeast", "humid_subtropical"),
    ("Sarasota", "FL", 27.3364, -82.5307, 58285, "Southeast", "humid_subtropical"),
    ("Naples", "FL", 26.1420, -81.7948, 22088, "Southeast", "tropical"),
    ("Key West", "FL", 24.5551, -81.7800, 25478, "Southeast", "tropical"),
    ("Pensacola", "FL", 30.4213, -87.2169, 54312, "Southeast", "humid_subtropical"),

    # Georgia
    ("Atlanta", "GA", 33.7490, -84.3880, 498715, "Southeast", "humid_subtropical"),
    ("Savannah", "GA", 32.0809, -81.0912, 147780, "Southeast", "humid_subtropical"),
    ("Augusta", "GA", 33.4735, -82.0105, 202081, "Southeast", "humid_subtropical"),
    ("Athens", "GA", 33.9519, -83.3576, 127315, "Southeast", "humid_subtropical"),
    ("Macon", "GA", 32.8407, -83.6324, 157346, "Southeast", "humid_subtropical"),

    # North Carolina
    ("Charlotte", "NC", 35.2271, -80.8431, 874579, "Southeast", "humid_subtropical"),
    ("Raleigh", "NC", 35.7796, -78.6382, 469298, "Southeast", "humid_subtropical"),
    ("Durham", "NC", 35.9940, -78.8986, 283506, "Southeast", "humid_subtropical"),
    ("Greensboro", "NC", 36.0726, -79.7920, 299035, "Southeast", "humid_subtropical"),
    ("Winston-Salem", "NC", 36.0999, -80.2442, 249545, "Southeast", "humid_subtropical"),
    ("Asheville", "NC", 35.5951, -82.5515, 94067, "Southeast", "humid_subtropical"),
    ("Wilmington", "NC", 34.2257, -77.9447, 123784, "Southeast", "humid_subtropical"),
    ("Chapel Hill", "NC", 35.9132, -79.0558, 61960, "Southeast", "humid_subtropical"),

    # South Carolina
    ("Charleston", "SC", 32.7765, -79.9311, 150227, "Southeast", "humid_subtropical"),
    ("Columbia", "SC", 34.0007, -81.0348, 136632, "Southeast", "humid_subtropical"),
    ("Greenville", "SC", 34.8526, -82.3940, 70635, "Southeast", "humid_subtropical"),
    ("Myrtle Beach", "SC", 33.6891, -78.8867, 35682, "Southeast", "humid_subtropical"),

    # Virginia
    ("Virginia Beach", "VA", 36.8529, -75.9780, 459470, "Southeast", "humid_subtropical"),
    ("Richmond", "VA", 37.5407, -77.4360, 226610, "Southeast", "humid_subtropical"),
    ("Norfolk", "VA", 36.8508, -76.2859, 244703, "Southeast", "humid_subtropical"),
    ("Arlington", "VA", 38.8816, -77.0910, 238643, "Southeast", "humid_subtropical"),
    ("Alexandria", "VA", 38.8048, -77.0469, 159467, "Southeast", "humid_subtropical"),
    ("Charlottesville", "VA", 38.0293, -78.4767, 47266, "Southeast", "humid_subtropical"),
    ("Roanoke", "VA", 37.2710, -79.9414, 100011, "Southeast", "humid_subtropical"),

    # Tennessee
    ("Nashville", "TN", 36.1627, -86.7816, 689447, "Southeast", "humid_subtropical"),
    ("Memphis", "TN", 35.1495, -90.0490, 633104, "Southeast", "humid_subtropical"),
    ("Knoxville", "TN", 35.9606, -83.9207, 190740, "Southeast", "humid_subtropical"),
    ("Chattanooga", "TN", 35.0456, -85.3097, 181099, "Southeast", "humid_subtropical"),

    # Alabama
    ("Birmingham", "AL", 33.5186, -86.8104, 200733, "Southeast", "humid_subtropical"),
    ("Montgomery", "AL", 32.3792, -86.3077, 200603, "Southeast", "humid_subtropical"),
    ("Huntsville", "AL", 34.7304, -86.5861, 215006, "Southeast", "humid_subtropical"),
    ("Mobile", "AL", 30.6954, -88.0399, 187041, "Southeast", "humid_subtropical"),

    # Mississippi
    ("Jackson", "MS", 32.2988, -90.1848, 153701, "Southeast", "humid_subtropical"),
    ("Gulfport", "MS", 30.3674, -89.0928, 72926, "Southeast", "humid_subtropical"),
    ("Biloxi", "MS", 30.3960, -88.8853, 46212, "Southeast", "humid_subtropical"),

    # Louisiana
    ("New Orleans", "LA", 29.9511, -90.0715, 383997, "Southeast", "humid_subtropical"),
    ("Baton Rouge", "LA", 30.4515, -91.1871, 227470, "Southeast", "humid_subtropical"),
    ("Shreveport", "LA", 32.5252, -93.7502, 187593, "Southeast", "humid_subtropical"),
    ("Lafayette", "LA", 30.2241, -92.0198, 126185, "Southeast", "humid_subtropical"),

    # Kentucky
    ("Louisville", "KY", 38.2527, -85.7585, 633045, "Southeast", "humid_subtropical"),
    ("Lexington", "KY", 38.0406, -84.5037, 322570, "Southeast", "humid_subtropical"),
    ("Bowling Green", "KY", 74067, -86.4436, 74067, "Southeast", "humid_subtropical"),

    # West Virginia
    ("Charleston", "WV", 38.3498, -81.6326, 48864, "Southeast", "humid_subtropical"),
    ("Huntington", "WV", 38.4192, -82.4452, 46842, "Southeast", "humid_subtropical"),
    ("Morgantown", "WV", 39.6295, -79.9559, 30855, "Southeast", "humid_subtropical"),

    # Arkansas
    ("Little Rock", "AR", 34.7465, -92.2896, 202591, "Southeast", "humid_subtropical"),
    ("Fayetteville", "AR", 36.0822, -94.1719, 93949, "Southeast", "humid_subtropical"),
    ("Fort Smith", "AR", 35.3859, -94.3985, 89142, "Southeast", "humid_subtropical"),

    # ===== MIDWEST =====
    # Illinois
    ("Chicago", "IL", 41.8781, -87.6298, 2746388, "Midwest", "humid_continental"),
    ("Aurora", "IL", 41.7606, -88.3201, 180542, "Midwest", "humid_continental"),
    ("Naperville", "IL", 41.7508, -88.1535, 149540, "Midwest", "humid_continental"),
    ("Rockford", "IL", 42.2711, -89.0940, 148655, "Midwest", "humid_continental"),
    ("Springfield", "IL", 39.7817, -89.6501, 114394, "Midwest", "humid_continental"),
    ("Peoria", "IL", 40.6936, -89.5890, 113150, "Midwest", "humid_continental"),
    ("Champaign", "IL", 40.1164, -88.2434, 88302, "Midwest", "humid_continental"),

    # Michigan
    ("Detroit", "MI", 42.3314, -83.0458, 639111, "Midwest", "humid_continental"),
    ("Grand Rapids", "MI", 42.9634, -85.6681, 198917, "Midwest", "humid_continental"),
    ("Ann Arbor", "MI", 42.2808, -83.7430, 123851, "Midwest", "humid_continental"),
    ("Lansing", "MI", 42.7325, -84.5555, 118210, "Midwest", "humid_continental"),
    ("Flint", "MI", 43.0125, -83.6875, 95943, "Midwest", "humid_continental"),
    ("Kalamazoo", "MI", 42.2917, -85.5872, 74262, "Midwest", "humid_continental"),
    ("Traverse City", "MI", 44.7631, -85.6206, 15569, "Midwest", "humid_continental"),

    # Ohio
    ("Columbus", "OH", 39.9612, -82.9988, 905748, "Midwest", "humid_continental"),
    ("Cleveland", "OH", 41.4993, -81.6944, 372624, "Midwest", "humid_continental"),
    ("Cincinnati", "OH", 39.1031, -84.5120, 309317, "Midwest", "humid_continental"),
    ("Toledo", "OH", 41.6528, -83.5379, 270871, "Midwest", "humid_continental"),
    ("Akron", "OH", 41.0814, -81.5190, 190469, "Midwest", "humid_continental"),
    ("Dayton", "OH", 39.7589, -84.1916, 140407, "Midwest", "humid_continental"),

    # Indiana
    ("Indianapolis", "IN", 39.7684, -86.1581, 887642, "Midwest", "humid_continental"),
    ("Fort Wayne", "IN", 41.0793, -85.1394, 270402, "Midwest", "humid_continental"),
    ("Evansville", "IN", 37.9716, -87.5711, 117298, "Midwest", "humid_continental"),
    ("South Bend", "IN", 41.6764, -86.2520, 103453, "Midwest", "humid_continental"),
    ("Bloomington", "IN", 39.1653, -86.5264, 79168, "Midwest", "humid_continental"),

    # Wisconsin
    ("Milwaukee", "WI", 43.0389, -87.9065, 577222, "Midwest", "humid_continental"),
    ("Madison", "WI", 43.0731, -89.4012, 269840, "Midwest", "humid_continental"),
    ("Green Bay", "WI", 44.5133, -88.0133, 107395, "Midwest", "humid_continental"),
    ("Eau Claire", "WI", 44.8113, -91.4985, 69421, "Midwest", "humid_continental"),

    # Minnesota
    ("Minneapolis", "MN", 44.9778, -93.2650, 429954, "Midwest", "humid_continental"),
    ("St. Paul", "MN", 44.9537, -93.0900, 311527, "Midwest", "humid_continental"),
    ("Rochester", "MN", 44.0121, -92.4802, 121395, "Midwest", "humid_continental"),
    ("Duluth", "MN", 46.7867, -92.1005, 90931, "Midwest", "humid_continental"),

    # Iowa
    ("Des Moines", "IA", 41.5868, -93.6250, 214133, "Midwest", "humid_continental"),
    ("Cedar Rapids", "IA", 41.9779, -91.6656, 137710, "Midwest", "humid_continental"),
    ("Iowa City", "IA", 41.6611, -91.5302, 74828, "Midwest", "humid_continental"),
    ("Davenport", "IA", 41.5236, -90.5776, 101724, "Midwest", "humid_continental"),

    # Missouri
    ("Kansas City", "MO", 39.0997, -94.5786, 508090, "Midwest", "humid_continental"),
    ("St. Louis", "MO", 38.6270, -90.1994, 301578, "Midwest", "humid_continental"),
    ("Springfield", "MO", 37.2090, -93.2923, 169176, "Midwest", "humid_continental"),
    ("Columbia", "MO", 38.9517, -92.3341, 126254, "Midwest", "humid_continental"),

    # Kansas
    ("Wichita", "KS", 37.6872, -97.3301, 397532, "Midwest", "humid_continental"),
    ("Overland Park", "KS", 38.9822, -94.6708, 197238, "Midwest", "humid_continental"),
    ("Kansas City", "KS", 39.1141, -94.6275, 156607, "Midwest", "humid_continental"),
    ("Topeka", "KS", 39.0473, -95.6752, 126587, "Midwest", "humid_continental"),
    ("Lawrence", "KS", 38.9717, -95.2353, 98193, "Midwest", "humid_continental"),

    # Nebraska
    ("Omaha", "NE", 41.2565, -95.9345, 486051, "Midwest", "humid_continental"),
    ("Lincoln", "NE", 40.8258, -96.6852, 291082, "Midwest", "humid_continental"),

    # North Dakota
    ("Fargo", "ND", 46.8772, -96.7898, 125990, "Midwest", "humid_continental"),
    ("Bismarck", "ND", 46.8083, -100.7837, 74112, "Midwest", "humid_continental"),

    # South Dakota
    ("Sioux Falls", "SD", 43.5460, -96.7313, 192517, "Midwest", "humid_continental"),
    ("Rapid City", "SD", 44.0805, -103.2310, 74703, "Mountain", "semi_arid"),

    # ===== SOUTHWEST =====
    # Texas
    ("Houston", "TX", 29.7604, -95.3698, 2320268, "Southwest", "humid_subtropical"),
    ("San Antonio", "TX", 29.4241, -98.4936, 1547253, "Southwest", "humid_subtropical"),
    ("Dallas", "TX", 32.7767, -96.7970, 1304379, "Southwest", "humid_subtropical"),
    ("Austin", "TX", 30.2672, -97.7431, 978908, "Southwest", "humid_subtropical"),
    ("Fort Worth", "TX", 32.7555, -97.3308, 918915, "Southwest", "humid_subtropical"),
    ("El Paso", "TX", 31.7619, -106.4850, 678815, "Southwest", "desert"),
    ("Arlington", "TX", 32.7357, -97.1081, 398854, "Southwest", "humid_subtropical"),
    ("Corpus Christi", "TX", 27.8006, -97.3964, 326586, "Southwest", "humid_subtropical"),
    ("Plano", "TX", 33.0198, -96.6989, 288061, "Southwest", "humid_subtropical"),
    ("Lubbock", "TX", 33.5779, -101.8552, 263930, "Southwest", "semi_arid"),
    ("Amarillo", "TX", 35.2220, -101.8313, 200393, "Southwest", "semi_arid"),
    ("Midland", "TX", 31.9973, -102.0779, 146038, "Southwest", "semi_arid"),

    # Arizona
    ("Phoenix", "AZ", 33.4484, -112.0740, 1680992, "Southwest", "desert"),
    ("Tucson", "AZ", 32.2226, -110.9747, 542629, "Southwest", "desert"),
    ("Mesa", "AZ", 33.4152, -111.8315, 518012, "Southwest", "desert"),
    ("Scottsdale", "AZ", 33.4942, -111.9261, 258069, "Southwest", "desert"),
    ("Tempe", "AZ", 33.4255, -111.9400, 191607, "Southwest", "desert"),
    ("Flagstaff", "AZ", 35.1983, -111.6513, 73964, "Southwest", "semi_arid"),
    ("Sedona", "AZ", 34.8697, -111.7610, 10336, "Southwest", "semi_arid"),

    # New Mexico
    ("Albuquerque", "NM", 35.0844, -106.6504, 564559, "Southwest", "semi_arid"),
    ("Las Cruces", "NM", 32.3199, -106.7637, 111385, "Southwest", "desert"),
    ("Santa Fe", "NM", 35.6870, -105.9378, 87505, "Southwest", "semi_arid"),
    ("Taos", "NM", 36.4072, -105.5731, 6193, "Southwest", "semi_arid"),

    # Oklahoma
    ("Oklahoma City", "OK", 35.4676, -97.5164, 681054, "Southwest", "humid_subtropical"),
    ("Tulsa", "OK", 36.1540, -95.9928, 413066, "Southwest", "humid_subtropical"),
    ("Norman", "OK", 35.2226, -97.4395, 128026, "Southwest", "humid_subtropical"),

    # ===== MOUNTAIN WEST =====
    # Colorado
    ("Denver", "CO", 39.7392, -104.9903, 727211, "Mountain", "semi_arid"),
    ("Colorado Springs", "CO", 38.8339, -104.8214, 478961, "Mountain", "semi_arid"),
    ("Aurora", "CO", 39.7294, -104.8319, 386261, "Mountain", "semi_arid"),
    ("Fort Collins", "CO", 40.5853, -105.0844, 169810, "Mountain", "semi_arid"),
    ("Boulder", "CO", 40.0150, -105.2705, 108250, "Mountain", "semi_arid"),
    ("Aspen", "CO", 39.1911, -106.8175, 7004, "Mountain", "semi_arid"),
    ("Vail", "CO", 39.6403, -106.3742, 5305, "Mountain", "semi_arid"),
    ("Telluride", "CO", 37.9375, -107.8123, 2607, "Mountain", "semi_arid"),
    ("Durango", "CO", 37.2753, -107.8801, 19115, "Mountain", "semi_arid"),

    # Utah
    ("Salt Lake City", "UT", 40.7608, -111.8910, 200567, "Mountain", "semi_arid"),
    ("Provo", "UT", 40.2338, -111.6585, 115919, "Mountain", "semi_arid"),
    ("Ogden", "UT", 41.2230, -111.9738, 87773, "Mountain", "semi_arid"),
    ("St. George", "UT", 37.0965, -113.5684, 95342, "Mountain", "desert"),
    ("Park City", "UT", 40.6461, -111.4980, 8467, "Mountain", "semi_arid"),
    ("Moab", "UT", 38.5733, -109.5498, 5366, "Mountain", "desert"),

    # Nevada
    ("Las Vegas", "NV", 36.1699, -115.1398, 651319, "Mountain", "desert"),
    ("Reno", "NV", 39.5296, -119.8138, 264165, "Mountain", "semi_arid"),
    ("Henderson", "NV", 36.0395, -114.9817, 320189, "Mountain", "desert"),

    # Idaho
    ("Boise", "ID", 43.6150, -116.2023, 235684, "Mountain", "semi_arid"),
    ("Nampa", "ID", 43.5407, -116.5635, 100200, "Mountain", "semi_arid"),
    ("Meridian", "ID", 43.6121, -116.3915, 117635, "Mountain", "semi_arid"),
    ("Idaho Falls", "ID", 43.4666, -112.0341, 64597, "Mountain", "semi_arid"),
    ("Coeur d'Alene", "ID", 47.6777, -116.7805, 54628, "Mountain", "semi_arid"),
    ("Sun Valley", "ID", 43.6969, -114.3517, 1786, "Mountain", "semi_arid"),

    # Montana
    ("Billings", "MT", 45.7833, -108.5007, 117116, "Mountain", "semi_arid"),
    ("Missoula", "MT", 46.8721, -113.9940, 75516, "Mountain", "semi_arid"),
    ("Bozeman", "MT", 45.6770, -111.0429, 53293, "Mountain", "semi_arid"),
    ("Helena", "MT", 46.5891, -112.0391, 34425, "Mountain", "semi_arid"),
    ("Whitefish", "MT", 48.4111, -114.3528, 8032, "Mountain", "semi_arid"),

    # Wyoming
    ("Cheyenne", "WY", 41.1400, -104.8202, 65132, "Mountain", "semi_arid"),
    ("Casper", "WY", 42.8666, -106.3131, 58210, "Mountain", "semi_arid"),
    ("Jackson", "WY", 43.4799, -110.7624, 10529, "Mountain", "semi_arid"),

    # ===== WEST COAST =====
    # California
    ("Los Angeles", "CA", 34.0522, -118.2437, 3979576, "West", "mediterranean"),
    ("San Diego", "CA", 32.7157, -117.1611, 1423851, "West", "mediterranean"),
    ("San Jose", "CA", 37.3382, -121.8863, 1013240, "West", "mediterranean"),
    ("San Francisco", "CA", 37.7749, -122.4194, 873965, "West", "mediterranean"),
    ("Fresno", "CA", 36.7378, -119.7871, 542107, "West", "semi_arid"),
    ("Sacramento", "CA", 38.5816, -121.4944, 513624, "West", "mediterranean"),
    ("Long Beach", "CA", 33.7701, -118.1937, 466742, "West", "mediterranean"),
    ("Oakland", "CA", 37.8044, -122.2712, 433031, "West", "mediterranean"),
    ("Bakersfield", "CA", 35.3733, -119.0187, 403455, "West", "semi_arid"),
    ("Anaheim", "CA", 33.8366, -117.9143, 350365, "West", "mediterranean"),
    ("Santa Ana", "CA", 33.7455, -117.8677, 310227, "West", "mediterranean"),
    ("Riverside", "CA", 33.9806, -117.3755, 314998, "West", "semi_arid"),
    ("Irvine", "CA", 33.6846, -117.8265, 307670, "West", "mediterranean"),
    ("Santa Barbara", "CA", 34.4208, -119.6982, 91350, "West", "mediterranean"),
    ("San Luis Obispo", "CA", 35.2828, -120.6596, 47536, "West", "mediterranean"),
    ("Santa Cruz", "CA", 36.9741, -122.0308, 65011, "West", "mediterranean"),
    ("Monterey", "CA", 36.6002, -121.8947, 28454, "West", "mediterranean"),
    ("Napa", "CA", 38.2975, -122.2869, 79068, "West", "mediterranean"),
    ("Palm Springs", "CA", 33.8303, -116.5453, 48518, "West", "desert"),
    ("Pasadena", "CA", 34.1478, -118.1445, 138699, "West", "mediterranean"),
    ("Berkeley", "CA", 37.8716, -122.2727, 124321, "West", "mediterranean"),
    ("Palo Alto", "CA", 37.4419, -122.1430, 68572, "West", "mediterranean"),

    # Oregon
    ("Portland", "OR", 45.5152, -122.6784, 652503, "West", "oceanic"),
    ("Salem", "OR", 44.9429, -123.0351, 175535, "West", "oceanic"),
    ("Eugene", "OR", 44.0521, -123.0868, 176654, "West", "oceanic"),
    ("Bend", "OR", 44.0582, -121.3153, 99178, "Mountain", "semi_arid"),
    ("Medford", "OR", 42.3265, -122.8756, 85824, "West", "mediterranean"),
    ("Ashland", "OR", 42.1946, -122.7095, 21607, "West", "mediterranean"),

    # Washington
    ("Seattle", "WA", 47.6062, -122.3321, 737015, "West", "oceanic"),
    ("Spokane", "WA", 47.6588, -117.4260, 222081, "Mountain", "semi_arid"),
    ("Tacoma", "WA", 47.2529, -122.4443, 219346, "West", "oceanic"),
    ("Vancouver", "WA", 45.6387, -122.6615, 190915, "West", "oceanic"),
    ("Bellevue", "WA", 47.6101, -122.2015, 151854, "West", "oceanic"),
    ("Olympia", "WA", 47.0379, -122.9007, 55605, "West", "oceanic"),
    ("Bellingham", "WA", 48.7519, -122.4787, 91482, "West", "oceanic"),

    # ===== ALASKA & HAWAII =====
    ("Anchorage", "AK", 61.2181, -149.9003, 291247, "Alaska", "subarctic"),
    ("Fairbanks", "AK", 64.8378, -147.7164, 32515, "Alaska", "subarctic"),
    ("Juneau", "AK", 58.3019, -134.4197, 32255, "Alaska", "oceanic"),

    ("Honolulu", "HI", 21.3069, -157.8583, 350964, "Hawaii", "tropical"),
    ("Hilo", "HI", 19.7074, -155.0885, 46594, "Hawaii", "tropical"),
    ("Kailua", "HI", 21.4022, -157.7394, 40514, "Hawaii", "tropical"),
    ("Maui (Kahului)", "HI", 20.8893, -156.4729, 29926, "Hawaii", "tropical"),
]

# Fix the Bowling Green entry (lat/lon were swapped)
REAL_CITIES = [(name, state, lat, lon, pop, region, climate)
               if name != "Bowling Green"
               else ("Bowling Green", "KY", 36.9685, -86.4436, 74067, "Southeast", "humid_subtropical")
               for name, state, lat, lon, pop, region, climate in REAL_CITIES]

# State metadata
STATE_INFO = {
    "no_income_tax": ["AK", "FL", "NV", "NH", "SD", "TN", "TX", "WA", "WY"],
    "low_property_tax": ["AL", "AZ", "CO", "HI", "LA", "MS", "NV", "SC", "WV", "WY"],
    "high_property_tax": ["CT", "IL", "NJ", "NH", "TX", "VT", "WI"],
}

# Climate data approximations by climate zone
CLIMATE_PROFILES = {
    "tropical": {
        "avg_temp_summer": (82, 90),
        "avg_temp_winter": (70, 78),
        "sunny_days": (240, 280),
        "annual_rainfall": (50, 65),
        "annual_snow": (0, 0),
    },
    "humid_subtropical": {
        "avg_temp_summer": (78, 92),
        "avg_temp_winter": (35, 55),
        "sunny_days": (200, 260),
        "annual_rainfall": (40, 60),
        "annual_snow": (0, 12),
    },
    "humid_continental": {
        "avg_temp_summer": (70, 85),
        "avg_temp_winter": (18, 35),
        "sunny_days": (160, 220),
        "annual_rainfall": (30, 50),
        "annual_snow": (30, 100),
    },
    "mediterranean": {
        "avg_temp_summer": (68, 85),
        "avg_temp_winter": (45, 58),
        "sunny_days": (260, 320),
        "annual_rainfall": (12, 25),
        "annual_snow": (0, 2),
    },
    "oceanic": {
        "avg_temp_summer": (62, 75),
        "avg_temp_winter": (35, 45),
        "sunny_days": (140, 180),
        "annual_rainfall": (35, 55),
        "annual_snow": (3, 15),
    },
    "semi_arid": {
        "avg_temp_summer": (75, 95),
        "avg_temp_winter": (25, 42),
        "sunny_days": (250, 310),
        "annual_rainfall": (10, 20),
        "annual_snow": (15, 60),
    },
    "desert": {
        "avg_temp_summer": (95, 108),
        "avg_temp_winter": (45, 62),
        "sunny_days": (290, 330),
        "annual_rainfall": (4, 12),
        "annual_snow": (0, 2),
    },
    "subarctic": {
        "avg_temp_summer": (55, 68),
        "avg_temp_winter": (-10, 18),
        "sunny_days": (150, 200),
        "annual_rainfall": (15, 25),
        "annual_snow": (60, 120),
    },
}


def generate_climate_data(climate_zone, lat):
    """Generate climate data based on climate zone."""
    profile = CLIMATE_PROFILES.get(climate_zone, CLIMATE_PROFILES["humid_continental"])
    return {
        "avg_temp_summer": np.random.uniform(*profile["avg_temp_summer"]),
        "avg_temp_winter": np.random.uniform(*profile["avg_temp_winter"]),
        "sunny_days": np.random.randint(*profile["sunny_days"]),
        "annual_rainfall": np.random.uniform(*profile["annual_rainfall"]),
        "annual_snow": np.random.uniform(*profile["annual_snow"]),
    }


def generate_cost_data(population, region, state):
    """Generate cost of living data."""
    base_index = 100

    if population > 1000000:
        pop_factor = np.random.uniform(1.15, 1.5)
    elif population > 500000:
        pop_factor = np.random.uniform(1.0, 1.3)
    elif population > 100000:
        pop_factor = np.random.uniform(0.9, 1.15)
    else:
        pop_factor = np.random.uniform(0.8, 1.05)

    region_factors = {
        "Northeast": 1.15, "West": 1.25, "Southeast": 0.92,
        "Midwest": 0.88, "Southwest": 0.95, "Mountain": 1.02,
        "Alaska": 1.3, "Hawaii": 1.5,
    }
    region_factor = region_factors.get(region, 1.0)

    cost_index = base_index * pop_factor * region_factor * np.random.uniform(0.92, 1.08)
    base_home_price = 350000
    median_home_price = base_home_price * (cost_index / 100) * np.random.uniform(0.85, 1.15)

    no_income_tax = state in STATE_INFO["no_income_tax"]
    state_income_tax = 0 if no_income_tax else np.random.uniform(3, 9.5)
    state_sales_tax = np.random.uniform(0, 9.5)

    if state in STATE_INFO["low_property_tax"]:
        property_tax = np.random.uniform(0.4, 1.0)
    elif state in STATE_INFO["high_property_tax"]:
        property_tax = np.random.uniform(1.8, 2.8)
    else:
        property_tax = np.random.uniform(0.9, 1.8)

    return {
        "cost_of_living_index": round(cost_index, 1),
        "median_home_price": int(median_home_price),
        "state_income_tax_rate": round(state_income_tax, 2),
        "state_sales_tax_rate": round(state_sales_tax, 2),
        "avg_property_tax_rate": round(property_tax, 2),
        "no_income_tax_state": no_income_tax,
    }


def generate_geography_data(name, region, climate_zone, lat, lon):
    """Generate geography data based on known city characteristics."""
    # Mountain cities
    mountain_cities = [
        "Denver", "Boulder", "Colorado Springs", "Fort Collins", "Aspen", "Vail", "Telluride", "Durango",
        "Salt Lake City", "Provo", "Park City", "Ogden",
        "Boise", "Sun Valley", "Coeur d'Alene",
        "Billings", "Missoula", "Bozeman", "Whitefish", "Helena",
        "Reno", "Flagstaff", "Sedona", "Taos", "Santa Fe", "Albuquerque",
        "Asheville", "Chattanooga", "Roanoke",
        "Spokane", "Bend", "Jackson", "Cheyenne", "Casper",
        "Anchorage", "Fairbanks", "Juneau",
    ]

    # Coastal cities
    coastal_cities = [
        "Miami", "Tampa", "St. Petersburg", "Fort Lauderdale", "Jacksonville", "Sarasota", "Naples", "Key West", "Pensacola",
        "San Diego", "Los Angeles", "Long Beach", "Santa Barbara", "San Luis Obispo", "Santa Cruz", "Monterey", "San Francisco", "Oakland",
        "Seattle", "Tacoma", "Olympia", "Bellingham", "Portland", "Astoria",
        "Boston", "Providence", "New York", "Jersey City", "Newark",
        "Savannah", "Charleston", "Myrtle Beach", "Wilmington", "Virginia Beach", "Norfolk",
        "New Orleans", "Mobile", "Biloxi", "Gulfport", "Corpus Christi",
        "Honolulu", "Hilo", "Kailua", "Maui (Kahului)",
        "Anchorage", "Juneau",
    ]

    # Lake cities
    lake_cities = [
        "Chicago", "Milwaukee", "Green Bay", "Madison",
        "Detroit", "Grand Rapids", "Traverse City", "Ann Arbor",
        "Cleveland", "Toledo", "Erie", "Buffalo", "Rochester", "Syracuse",
        "Minneapolis", "St. Paul", "Duluth",
        "Salt Lake City", "Reno", "Coeur d'Alene",
        "Austin", "Orlando", "Tampa",
    ]

    has_mountains = name in mountain_cities or region == "Mountain"
    has_ocean = name in coastal_cities
    has_lakes = name in lake_cities or region in ["Midwest"]
    has_desert = climate_zone in ["desert"] or name in ["Phoenix", "Tucson", "Las Vegas", "Palm Springs", "El Paso"]

    # Ski resort distance
    if has_mountains and region == "Mountain":
        ski_distance = np.random.uniform(15, 80)
    elif has_mountains:
        ski_distance = np.random.uniform(40, 150)
    elif region in ["West", "Northeast"] and lat > 38:
        ski_distance = np.random.uniform(80, 250)
    else:
        ski_distance = np.random.uniform(300, 800)

    return {
        "has_mountains": has_mountains,
        "has_ocean": has_ocean,
        "has_lakes": has_lakes,
        "has_desert": has_desert,
        "ski_resort_distance_miles": round(ski_distance, 0),
        "ski_resorts_within_100mi": max(0, int(8 - ski_distance / 15)) if ski_distance < 120 else 0,
        "hiking_trails_count": np.random.randint(50, 400) if has_mountains else np.random.randint(10, 100),
        "mountain_biking_trails": np.random.randint(20, 150) if has_mountains else np.random.randint(5, 40),
        "rock_climbing_areas_nearby": np.random.randint(5, 25) if has_mountains else np.random.randint(0, 5),
        "swimming_access": "ocean" if has_ocean else ("lake" if has_lakes else "pool"),
        "national_parks_within_100mi": np.random.randint(1, 5) if has_mountains else np.random.randint(0, 2),
        "state_parks_nearby": np.random.randint(3, 15),
        "camping_areas_count": np.random.randint(20, 80) if has_mountains else np.random.randint(5, 30),
    }


def generate_transportation_data(name, population, region):
    """Generate transportation data."""
    hub_cities = [
        "New York", "Los Angeles", "Chicago", "Dallas", "Denver", "Atlanta",
        "San Francisco", "Seattle", "Miami", "Phoenix", "Houston", "Boston",
        "Minneapolis", "Detroit", "Philadelphia", "Charlotte", "Las Vegas",
        "Orlando", "Newark", "Salt Lake City", "Washington"
    ]

    is_hub = name in hub_cities
    is_major = population > 400000

    return {
        "nearest_major_airport": f"{name} Airport",
        "airport_distance_miles": np.random.uniform(5, 25) if is_major else np.random.uniform(15, 60),
        "is_airline_hub": is_hub,
        "direct_flight_destinations_count": int(
            np.random.uniform(150, 300) if is_hub else
            np.random.uniform(50, 150) if population > 500000 else
            np.random.uniform(15, 60) if population > 100000 else
            np.random.uniform(5, 20)
        ),
        "walkability_score": int(
            np.random.uniform(70, 95) if population > 500000 else
            np.random.uniform(45, 75) if population > 100000 else
            np.random.uniform(25, 55)
        ),
        "transit_score": int(
            np.random.uniform(60, 90) if population > 500000 else
            np.random.uniform(30, 60) if population > 100000 else
            np.random.uniform(10, 35)
        ),
    }


def generate_education_data(name, population):
    """Generate education data."""
    university_cities = [
        "Boston", "Cambridge", "New Haven", "Providence", "Ithaca", "Princeton",
        "Ann Arbor", "Madison", "Austin", "Berkeley", "Palo Alto", "Stanford",
        "Chapel Hill", "Durham", "Charlottesville", "Athens", "Gainesville",
        "Boulder", "Eugene", "Iowa City", "Bloomington", "State College",
        "Columbia", "Lawrence", "Norman", "Tucson", "Tempe",
    ]

    has_major_uni = name in university_cities or (population > 100000 and np.random.random() < 0.4)

    return {
        "school_quality_score": np.random.uniform(5, 9.5),
        "university_count": np.random.randint(1, 8) if population > 100000 else np.random.randint(0, 2),
        "has_major_university": has_major_uni,
        "community_college_nearby": population > 20000 or np.random.random() < 0.8,
        "avg_school_rating": np.random.uniform(5.5, 8.5),
    }


def generate_culture_data(population):
    """Generate culture and entertainment data."""
    is_large = population > 500000
    is_medium = population > 100000

    return {
        "pro_sports_teams": np.random.randint(2, 7) if is_large else (1 if is_medium and np.random.random() < 0.25 else 0),
        "performing_arts_venues": np.random.randint(15, 80) if is_large else np.random.randint(3, 20),
        "broadway_tour_stop": is_large or (is_medium and np.random.random() < 0.25),
        "museums_count": np.random.randint(25, 150) if is_large else np.random.randint(5, 30),
        "concert_venue_capacity": np.random.randint(15000, 70000) if is_large else np.random.randint(2000, 15000),
    }


def generate_financial_health(population, region, cost_index):
    """Generate financial health data."""
    base_income = 65000
    income = base_income * (cost_index / 100) * np.random.uniform(0.88, 1.12)

    return {
        "median_household_income": int(income),
        "unemployment_rate": np.random.uniform(2.8, 6.5),
        "poverty_rate": np.random.uniform(6, 18),
        "job_growth_rate": np.random.uniform(-1, 4.5),
        "median_home_value_growth": np.random.uniform(-3, 12),
        "municipal_bond_rating": np.random.choice(["AAA", "AA+", "AA", "AA-", "A+", "A"]),
        "economic_diversity_index": np.random.uniform(0.4, 0.85),
    }


def generate_safety_data(population):
    """Generate safety data."""
    base_crime = 25
    if population > 500000:
        crime_factor = np.random.uniform(1.1, 1.7)
    elif population > 100000:
        crime_factor = np.random.uniform(0.85, 1.35)
    else:
        crime_factor = np.random.uniform(0.6, 1.1)

    crime_rate = base_crime * crime_factor

    return {
        "crime_rate_per_1000": round(crime_rate, 1),
        "violent_crime_rate": round(crime_rate * np.random.uniform(0.15, 0.3), 1),
        "property_crime_rate": round(crime_rate * np.random.uniform(0.65, 0.85), 1),
    }


def generate_city_data(city_tuple):
    """Generate complete data for a single city."""
    name, state, lat, lon, population, region, climate_zone = city_tuple

    metro_factor = np.random.uniform(1.8, 4.5) if population > 100000 else np.random.uniform(1.2, 2.0)
    metro_pop = int(population * metro_factor)

    climate = generate_climate_data(climate_zone, lat)
    cost = generate_cost_data(population, region, state)
    geography = generate_geography_data(name, region, climate_zone, lat, lon)
    transport = generate_transportation_data(name, population, region)
    education = generate_education_data(name, population)
    culture = generate_culture_data(population)
    financial = generate_financial_health(population, region, cost["cost_of_living_index"])
    safety = generate_safety_data(population)

    industries = np.random.choice([
        "Technology", "Healthcare", "Finance", "Manufacturing",
        "Education", "Tourism", "Government", "Energy", "Retail", "Agriculture"
    ], size=np.random.randint(2, 4), replace=False).tolist()

    return {
        "city_id": f"{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}_{state.lower()}",
        "name": name,
        "state": state,
        "lat": lat,
        "lon": lon,
        "population": population,
        "metro_pop": metro_pop,
        "region": region,
        **climate,
        **cost,
        **geography,
        **transport,
        **education,
        **culture,
        **financial,
        **safety,
        "major_industries": ",".join(industries),
    }


def build_database():
    """Build the city database with real US cities only."""
    print("Building city database with REAL US cities only...")
    print(f"Total cities: {len(REAL_CITIES)}")

    city_data = []
    for i, city in enumerate(REAL_CITIES):
        city_data.append(generate_city_data(city))

    df = pd.DataFrame(city_data)

    output_path = DATA_DIR / "cities.parquet"
    df.to_parquet(output_path, index=False)
    print(f"\nSaved {len(df)} real cities to {output_path}")

    print("\nDatabase Summary:")
    print(f"  Total cities: {len(df)}")
    print(f"  States covered: {df['state'].nunique()}")
    print(f"  Population range: {df['population'].min():,} - {df['population'].max():,}")
    print(f"  Regions: {df['region'].unique().tolist()}")

    return df


if __name__ == "__main__":
    df = build_database()
    print("\nSample cities:")
    print(df[["name", "state", "population", "region"]].head(20).to_string())
