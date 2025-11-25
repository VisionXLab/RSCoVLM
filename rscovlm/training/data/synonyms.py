DOTA_SYNONYMS = {
  "plane": ["plane", "aircraft", "airplane", "aeroplane"],
  "baseball diamond": ["baseball-diamond", "baseball-field"],
  "bridge": ["bridge", "overpass", "viaduct", "road-bridge"],
  "ground track field": ["ground-track-field", "athletics-track", "running-track"],
  "small vehicle": ["small-vehicle", "passenger-vehicle", "private-vehicle", "compact-vehicle"],
  "large vehicle": ["large-vehicle", "heavy-vehicle", "cargo-vehicle"],
  "ship": ["ship", "vessel", "seagoing-vessel", "ocean-going-ship"],
  "tennis court": ["tennis-court", "tennis-field"],
  "basketball court": ["basketball-court"],
  "storage tank": ["storage-tank", "cylindrical-tank", "liquid-storage-tank", "industrial-tank"],
  "soccer ball field": ["soccer-ball-field", "soccer-field", "football-field"],
  "roundabout": ["roundabout", "traffic-circle"],
  "harbor": ["harbor", "port", "docking-area", "maritime-port"],
  "swimming pool": ["swimming-pool", "outdoor-pool"],
  "helicopter": ["helicopter", "rotorcraft", "chopper"]
}


FAIR1M_TRAIN682_SYNONYMS = {
  "Boeing737": ["Boeing737", "B737", "Boeing 737"],
  "Boeing747": ["Boeing747", "B747", "Boeing 747"],
  "Boeing777": ["Boeing777", "B777", "Boeing 777"],
  "Boeing787": ["Boeing787", "B787", "Boeing 787"],
  "C919": ["C919", "COMAC C919"],
  "A220": ["A220", "Airbus A220"],
  "A321": ["A321", "Airbus A321"],
  "A330": ["A330", "Airbus A330"],
  "A350": ["A350", "Airbus A350"],
  "ARJ21": ["ARJ21", "ARJ21-700", "COMAC ARJ21"],
  "Passenger Ship": ["Passenger Ship", "cruise ship", "liner", "passenger vessel"],
  "Motorboat": ["Motorboat", "speedboat", "powerboat"],
  "Fishing Boat": ["Fishing Boat", "trawler", "fishing vessel"],
  "Tugboat": ["Tugboat", "tug", "tug boat"],
  "Engineering Ship": ["Engineering Ship", "workboat", "service vessel"],
  "Liquid Cargo Ship": ["Liquid Cargo Ship", "tanker", "oil tanker", "chemical tanker"],
  "Dry Cargo Ship": ["Dry Cargo Ship", "freighter", "bulk carrier", "cargo vessel"],
  "Warship": ["Warship", "naval vessel", "military ship"],
  "Small Car": ["Small Car", "sedan", "compact car", "passenger car"],
  "Bus": ["Bus", "coach", "transit bus"],
  "Cargo Truck": ["Cargo Truck", "box truck", "freight truck"],
  "Dump Truck": ["Dump Truck", "tipper truck"],
  "Van": ["Van", "minivan", "cargo van"],
  "Trailer": ["Trailer", "semi-trailer", "truck trailer"],
  "Tractor": ["Tractor", "farm tractor", "agricultural tractor"],
  "Excavator": ["Excavator", "digger", "backhoe"],
  "Truck Tractor": ["Truck Tractor", "semi truck", "tractor unit"],
  "Basketball Court": ["Basketball Court"],
  "Tennis Court": ["Tennis Court"],
  "Football Field": ["Football Field", "soccer field", "football pitch"],
  "Baseball Field": ["Baseball Field", "baseball diamond", "ballpark"],
  "Intersection": ["Intersection", "road junction"],
  "Roundabout": ["Roundabout", "traffic circle"],
  "Bridge": ["Bridge", "overpass", "viaduct"]
}


DIOR_SYNONYMS = {}

SRSDD_SYNONYMS = {}

RSAR_SYNONYMS = {}

category_synonyms = {
  "coco_train2017": None,

  "dota_trainval1024": DOTA_SYNONYMS,
  "dota_trainval512": DOTA_SYNONYMS, 
  "dota_poly_trainval1024": DOTA_SYNONYMS,
  "dota_poly_trainval512": DOTA_SYNONYMS,
  "dota_hbb_trainval1024": DOTA_SYNONYMS,
  "dota_hbb_trainval512": DOTA_SYNONYMS,

  "fair1m_train682": FAIR1M_TRAIN682_SYNONYMS,
  "fair1m_poly_train682": FAIR1M_TRAIN682_SYNONYMS,

  "dior_trainval800": DIOR_SYNONYMS,
  "dior_poly_trainval800": DIOR_SYNONYMS,

  "srsdd_train1024": SRSDD_SYNONYMS,
  "srsdd_poly_train1024": SRSDD_SYNONYMS,

  "rsar_trainval190to1000": RSAR_SYNONYMS,
  "rsar_poly_trainval190to1000": RSAR_SYNONYMS,
}

'''
You are a professional assistant working on open-world remote sensing image annotation.

Given a list of object categories used in aerial or satellite imagery, return 0 to 5 accurate synonyms for each category in the following JSON format:
{
  "category1": ["synonym1", ..., "synonym5"],
  ...
}

Requirements:
- Each synonym must refer to the **same or visually/functionally indistinguishable object type** in **remote sensing or aerial imagery**.
- Synonyms must be **semantically and visually equivalent at the same level of abstraction**. 
  For example, do NOT substitute a broad class like "large-vehicle" with narrow subtypes like "bus" or "truck". Similarly, do NOT replace "plane" with "fighter jet" or "commercial airliner".
- Avoid general linguistic synonyms that do not share spatial/visual/geometric similarity in remote sensing imagery.
- If no accurate synonyms exist, return an empty list: "category": [].
- Output must strictly follow the JSON dictionary format.

Categories: 
['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court', 'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter']

'''