#!/usr/bin/env python3.8

"""Calculate trade routes for GURPS Traveller: Far Trader"""

# Needed for circular class references in mypy.
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import defaultdict
from heapq import heappush, heappop
import math
import os
import shutil
from sys import maxsize
import tempfile
from typing import Dict, List, Optional, Set, Tuple
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import numpy
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall


starport_traveller_to_gurps = {
    "A": "V",
    "B": "IV",
    "C": "III",
    "D": "II",
    "E": "I",
    "X": "0",
}


tech_level_traveller_to_gurps = {
    0: 2,  # actually 1-3
    1: 4,
    2: 5,
    3: 5,
    4: 5,
    5: 6,
    6: 6,
    7: 7,
    8: 8,
    9: 9,
    10: 9,
    11: 9,
    12: 10,
    13: 10,
    14: 11,
    15: 12,
    16: 13,
}


# Global to allow any sector to find other sectors
location_to_sector = {}  # type: Dict[Tuple[int, int], Sector]

# Global to allow any world to find other worlds
abs_coords_to_world = {}  # type: Dict[Tuple[float, float], World]

# Global to have consistent indexes for navigable path computations
sorted_worlds = []  # type: List[World]

# Global so we only have to compute it once
navigable_dist = {}  # type: Dict[Tuple[World, World], int]

# Global so we only have to compute it once
predecessors = numpy.zeros((1, 1))  # type: numpy.ndarray


def download_sector_data(data_dir: str, sector_names: List[str]) -> None:
    for sector_name in sector_names:
        sector_data_filename = sector_name + ".sec"
        data_path = os.path.join(data_dir, sector_data_filename)
        sector_xml_filename = sector_name + ".xml"
        metadata_path = os.path.join(data_dir, sector_xml_filename)
        escaped_sector = urllib.parse.quote(sector_name)
        if not os.path.exists(data_path):
            url = f"https://travellermap.com/data/{escaped_sector}"
            response = urllib.request.urlopen(url)
            data = response.read()
            with open(data_path, "wb") as fil:
                fil.write(data)
        if not os.path.exists(metadata_path):
            url = f"https://travellermap.com/data/{escaped_sector}/metadata"
            response = urllib.request.urlopen(url)
            data = response.read()
            with open(metadata_path, "wb") as fil:
                fil.write(data)


def parse_header_and_separator(
    header: str, separator: str
) -> Dict[str, Tuple[int, int]]:
    """Parse header and separator and return {field: (start, end)}"""
    headers = header.split()
    separators = separator.split()
    field_to_start_end = {}  # type: Dict[str, Tuple[int, int]]
    column = 0
    for ii, hyphens in enumerate(separators):
        field = headers[ii]
        start = column
        width = len(hyphens)
        end = column + width
        field_to_start_end[field] = (start, end)
        column += width + 1
    return field_to_start_end


def same_allegiance(allegiance1: str, allegiance2: str) -> bool:
    if allegiance1 != allegiance2:
        return False
    if allegiance1.startswith("Na") or allegiance1.startswith("Cs"):
        # non-aligned worlds and client states with the same code are not
        # necessarily the same allegiance
        return False
    return True


def worlds_by_wtn() -> List[Tuple[float, World]]:
    """Must be run after all worlds are built."""
    wtn_worlds = []
    for world in sorted_worlds:
        wtn_worlds.append((world.wtn, world))
    wtn_worlds.sort()
    wtn_worlds.reverse()
    return wtn_worlds


def populate_navigable_distances() -> None:
    """Find minimum distances between all worlds using the Floyd-Warshall
    algorithm.

    Only use 1- and 2-hex jumps, except along xboat routes.

    Must be run after all neighbors are built.
    """
    global sorted_worlds
    sorted_worlds = sorted(abs_coords_to_world.values())
    index_to_world = {}
    for ii, world in enumerate(sorted_worlds):
        world.index = ii
        index_to_world[ii] = world
    nd = numpy.full((len(sorted_worlds), len(sorted_worlds)), maxsize)
    for ii, world in enumerate(sorted_worlds):
        for neighbor1 in world.neighbors1:
            nd[ii][neighbor1.index] = 1
        for neighbor2 in world.neighbors2:
            nd[ii][neighbor2.index] = 2
        for neighbor in world.xboat_routes:
            nd[ii][neighbor.index] = world.straight_line_distance(neighbor)
        nd[ii][ii] = 0
    global predecessors
    dist_matrix, predecessors = floyd_warshall(
        nd, directed=False, return_predecessors=True
    )
    global navigable_dist
    for yy, row in enumerate(dist_matrix):
        world1 = index_to_world[yy]
        for xx, dist in enumerate(row):
            world2 = index_to_world[xx]
            navigable_dist[world1, world2] = dist


def populate_trade_routes() -> None:
    """Fill in main_routes, minor_routes, and feeder_routes.

    This must be called after all Sectors and Worlds are mostly built.
    rules say: main: 10+  feeder: 9-9.5  minor: 8-8.5
    wiki says: blue major 12, cyan main 11, green intermediate 10,
               yellow feeder 9, red minor 8, no line 1-7
    """
    # TODO Merge trade routes together
    # TODO Mark the intermediate planets on the routes
    wtn_worlds = worlds_by_wtn()
    for ii, (wtn1, world1) in enumerate(wtn_worlds):
        for jj in range(ii + 1, len(wtn_worlds)):
            (wtn2, world2) = wtn_worlds[jj]
            btn = world1.btn(world2)
            if btn >= 10:
                world1.main_routes.add(world2)
                world2.main_routes.add(world1)
            elif btn >= 9:
                world1.feeder_routes.add(world2)
                world2.feeder_routes.add(world1)
            elif btn >= 8:
                world1.minor_routes.add(world2)
                world2.minor_routes.add(world1)


class World:
    sector: Sector
    hex_: str
    name: str
    uwp: str
    trade_classifications: Set[str]
    importance: int
    economic: str
    cultural: str
    nobles: str
    bases: Set[str]
    zone: str
    pbg: str
    worlds: int
    allegiance: str
    stars: List[str]
    xboat_routes: Set[World]
    main_routes: Set[World]
    feeder_routes: Set[World]
    minor_routes: Set[World]
    neighbors1: Set[World]
    neighbors2: Set[World]
    neighbors3: Set[World]
    index: int

    def __init__(
        self,
        line: str,
        field_to_start_end: Dict[str, Tuple[int, int]],
        sector: Sector,
    ) -> None:
        self.sector = sector
        self.bases = set()
        self.stars = []
        self.trade_classifications = set()
        self.xboat_routes = set()
        self.main_routes = set()
        self.feeder_routes = set()
        self.minor_routes = set()
        self.neighbors1 = set()
        self.neighbors2 = set()
        self.neighbors3 = set()
        for field, (start, end) in field_to_start_end.items():
            value = line[start:end]
            if field == "Hex":
                self.hex_ = value
            elif field == "Name":
                self.name = value.strip()
            elif field == "UWP":
                self.uwp = value
            elif field == "Remarks":
                self.trade_classifications = set(value.strip().split())
            elif field == "{Ix}":
                self.importance = int(value.strip("{} "))
            elif field == "(Ex)":
                self.economic = value.strip("()")
            elif field == "[Cx]":
                self.cultural = value.strip("[]")
            elif field == "N":
                self.nobles = value.strip()
            elif field == "B":
                for ch in value.strip():
                    self.bases.add(ch)
            elif field == "Z":
                self.zone = value.strip()
                if self.zone == "-":
                    self.zone = "G"
            elif field == "PBG":
                self.pbg = value.strip()
            elif field == "W":
                self.worlds = int(value.strip())
            elif field == "A":
                self.allegiance = value
            elif field == "Stellar":
                stars = value.strip().split()
                ii = 0
                while ii < len(stars):
                    star = stars[ii]
                    if star in {"BD", "D"}:
                        self.stars.append(star)
                        ii += 1
                    else:
                        self.stars.append(star + " " + stars[ii + 1])
                        ii += 2
        global abs_coords_to_world
        abs_coords_to_world[self.abs_coords] = self

    def populate_neighbors(self) -> None:
        """Find and cache all neighbors within 3 hexes.

        This must be run after all Sectors and Worlds are mostly initialized.
        """
        (x, y) = self.abs_coords
        xx = x - 3
        while xx <= x + 3:
            yy = y - 3
            while yy <= y + 3:
                world = abs_coords_to_world.get((xx, yy))
                if world is not None and world != self:
                    distance = self.straight_line_distance(world)
                    if distance == 1:
                        self.neighbors1.add(world)
                    elif distance == 2:
                        self.neighbors2.add(world)
                    elif distance == 3:
                        self.neighbors3.add(world)
                yy += 0.5
            xx += 1

    def __str__(self) -> str:
        return "World: " + self.name

    def __repr__(self) -> str:
        return "World: " + self.name

    def __eq__(self, other):
        if other is None:
            return False
        x1, y1 = self.abs_coords
        x2, y2 = other.abs_coords
        return x1 == x2 and y1 == y2

    def __hash__(self):
        x1, y1 = self.abs_coords
        return hash(x1) + hash(y1)

    # Impose consistent ordering to make paths predictable.
    def __lt__(self, other):
        x1, y1 = self.abs_coords
        x2, y2 = other.abs_coords
        if x1 < x2:
            return True
        elif y1 < y2:
            return True
        else:
            return False

    @property
    def starport(self) -> str:
        return self.uwp[0]

    @property
    def g_starport(self) -> str:
        return starport_traveller_to_gurps[self.uwp[0]]

    @property
    def size(self) -> str:
        return self.uwp[1]

    @property
    def atmosphere(self) -> str:
        return self.uwp[2]

    @property
    def hydrosphere(self) -> str:
        return self.uwp[3]

    @property
    def population(self) -> str:
        return self.uwp[4]

    @property
    def government(self) -> str:
        return self.uwp[5]

    @property
    def law_level(self) -> str:
        return self.uwp[6]

    @property
    def tech_level(self) -> str:
        return self.uwp[8]

    @property
    def g_tech_level(self) -> int:
        tech_level_int = int(self.tech_level, 17)
        return tech_level_traveller_to_gurps[tech_level_int]

    @property
    def uwtn(self) -> float:
        tl_mod = (self.g_tech_level // 3) / 2 - 0.5
        pop_mod = int(self.population, 16) / 2
        return tl_mod + pop_mod

    @property
    def wtn_port_modifier(self) -> float:
        table = {
            (7, "V"): 0.0,
            (7, "IV"): -1.0,
            (7, "III"): -1.5,
            (7, "II"): -2.0,
            (7, "I"): -2.5,
            (7, "0"): -5.0,
            (6, "V"): 0.0,
            (6, "IV"): -0.5,
            (6, "III"): -1.0,
            (6, "II"): -1.5,
            (6, "I"): -2.0,
            (6, "0"): -4.5,
            (5, "V"): 0.0,
            (5, "IV"): 0.0,
            (5, "III"): -0.5,
            (5, "II"): -1.0,
            (5, "I"): -1.5,
            (5, "0"): -4.0,
            (4, "V"): 0.5,
            (4, "IV"): 0.0,
            (4, "III"): 0.0,
            (4, "II"): -0.5,
            (4, "I"): -1.0,
            (4, "0"): -3.5,
            (3, "V"): 0.5,
            (3, "IV"): 0.5,
            (3, "III"): 0.0,
            (3, "II"): 0.0,
            (3, "I"): -0.5,
            (3, "0"): -3.0,
            (2, "V"): 1.0,
            (2, "IV"): 0.5,
            (2, "III"): 0.5,
            (2, "II"): 0.0,
            (2, "I"): 0.0,
            (2, "0"): -2.5,
            (1, "V"): 1.0,
            (1, "IV"): 1.0,
            (1, "III"): 0.5,
            (1, "II"): 0.0,
            (1, "I"): 0.0,
            (1, "0"): 0.0,
            (0, "V"): 1.5,
            (0, "IV"): 1.0,
            (0, "III"): 1.0,
            (0, "II"): 0.5,
            (0, "I"): 0.5,
            (0, "0"): 0.0,
        }
        iuwtn = max(0, int(self.uwtn))
        return table[(iuwtn, self.g_starport)]

    @property
    def wtn(self) -> float:
        return self.uwtn + self.wtn_port_modifier

    def wtcm(self, other: World) -> float:
        result = 0.0
        if "Ag" in self.trade_classifications:
            if (
                "Ex" in other.trade_classifications
                or "Na" in other.trade_classifications
            ):
                result += 0.5
        elif "Ag" in other.trade_classifications:
            if (
                "Ex" in self.trade_classifications
                or "Na" in self.trade_classifications
            ):
                result += 0.5
        if "In" in self.trade_classifications:
            if "Ni" in other.trade_classifications:
                result += 0.5
        elif "In" in other.trade_classifications:
            if "Ni" in self.trade_classifications:
                result += 0.5
        if not same_allegiance(self.allegiance, other.allegiance):
            result -= 0.5
        return result

    @property
    def abs_coords(self) -> Tuple[float, float]:
        hex_ = self.hex_
        location = self.sector.location
        x = int(hex_[:2]) + 32 * location[0]
        y = int(hex_[2:]) + 40 * location[1] + 0.5 * (x & 1 == 0)
        return (x, y)

    def straight_line_distance(self, other: World) -> int:
        """Return the shortest distance in hexes between the worlds"""
        (x1, y1) = self.abs_coords
        (x2, y2) = other.abs_coords
        xdelta = abs(x2 - x1)
        ydelta = max(0, abs(y2 - y1) - xdelta / 2)
        return math.floor(xdelta + ydelta)

    def navigable_distance(self, other: World) -> Optional[int]:
        """Return the length of the shortest navigable path from self to other.

        This uses jump-4 only along Xboat routes, and jump-2 otherwise.
        If it's not reachable, return None.
        This can only be called after populate_navigable_distances() runs.
        """
        dist = navigable_dist[self, other]
        if dist >= maxsize:
            return None
        return dist

    def navigable_path(self, other: World) -> Optional[List[World]]:
        """Return the shortest navigable path from self to other.

        If it's not reachable, return None.
        The path should include other but not self.
        This uses jump-4 only along Xboat routes, and jump-2 otherwise.
        This can only be called after populate_navigable_distances() runs.
        """
        if self == other:
            return []
        if self.navigable_distance(other) is None:
            return None
        world2 = self
        path = []
        while world2 != other:
            index = predecessors[other.index][world2.index]
            world2 = sorted_worlds[index]
            path.append(world2)
        return path

    def distance_modifier(self, other: World) -> float:
        distance = self.navigable_distance(other)
        if distance is None:
            return maxsize
        table = [1, 2, 5, 9, 19, 29, 59, 99, 199, 299, 599, 999, maxsize]
        index = bisect_left(table, distance)
        return index / 2

    def btn(self, other: World) -> float:
        min_wtn = min(self.wtn, other.wtn)
        base_btn = self.wtn + other.wtn + self.wtcm(other)
        btn = base_btn - self.distance_modifier(other)
        # The rules don't say that BTN can't be negative, but it seems more
        # reasonable not to go below zero.
        return max(0, min(btn, min_wtn + 5))

    def passenger_btn(self, other: World) -> float:
        min_wtn = min(self.wtn, other.wtn)
        base_btn = self.wtn + other.wtn + self.wtcm(other)
        pbtn = base_btn - self.distance_modifier(other)
        for world in [self, other]:
            if "Ri" in world.trade_classifications:
                pbtn += 0.5
            if "Cp" in world.trade_classifications:
                pbtn += 0.5
            if "Cs" in world.trade_classifications:
                pbtn += 1.0
        # The rules don't say that PBTN can't be negative, but it seems more
        # reasonable not to go below zero.
        return max(0, min(pbtn, min_wtn + 5))


class Sector:
    names: List[str]
    abbreviation: str
    location: Tuple[int, int]
    subsector_letter_to_name: Dict[str, str]
    allegiance_code_to_name: Dict[str, str]
    hex_to_world: Dict[str, World]

    def __init__(self, data_dir: str, sector_name: str) -> None:
        self.names = []
        self.subsector_letter_to_name = {}
        self.allegiance_code_to_name = {}
        self.hex_to_world = {}

        self.parse_xml_metadata(data_dir, sector_name)
        self.parse_column_data(data_dir, sector_name)

    def __str__(self) -> str:
        return "Sector: " + self.name

    def __repr__(self) -> str:
        return "Sector: " + self.name

    def parse_xml_metadata(self, data_dir: str, sector_name: str):
        xml_path = os.path.join(data_dir, sector_name + ".xml")
        tree = ET.parse(xml_path)
        root_element = tree.getroot()
        self.abbreviation = root_element.attrib["Abbreviation"]
        x = maxsize
        x_element = root_element.find("X")
        if x_element is not None:
            x_text = x_element.text
            if x_text:
                x = int(x_text)
        y = maxsize
        y_element = root_element.find("Y")
        if y_element is not None:
            y_text = y_element.text
            if y_text:
                y = int(y_text)
        self.location = (x, y)
        name_elements = root_element.findall("Name")
        for name_element in name_elements:
            if "Lang" not in name_element.attrib:
                if name_element.text:
                    self.names.append(name_element.text)
            else:
                if name_element.text:
                    self.names.append(
                        name_element.text
                        + " ("
                        + name_element.attrib["Lang"]
                        + ")"
                    )
        subsectors_element = root_element.find("Subsectors")
        if subsectors_element:
            for subsector_element in subsectors_element.findall("Subsector"):
                letter = subsector_element.attrib.get("Index")
                subsector_name = subsector_element.text
                if letter and subsector_name:
                    self.subsector_letter_to_name[letter] = subsector_name
        allegiances_element = root_element.find("Allegiances")
        if allegiances_element:
            for allegiance_element in allegiances_element.findall(
                "Allegiance"
            ):
                allegiance_code = allegiance_element.attrib["Code"]
                allegiance_name = allegiance_element.text
                if allegiance_name:
                    self.allegiance_code_to_name[
                        allegiance_code
                    ] = allegiance_name
        global location_to_sector
        # Set this last, after the sector is as fully built as possible.
        location_to_sector[self.location] = self

    def parse_column_data(self, data_dir: str, sector_name: str):
        sector_data_filename = sector_name + ".sec"
        data_path = os.path.join(data_dir, sector_data_filename)
        with open(data_path) as fil:
            for line in fil:
                line = line.strip()
                if not line:
                    continue
                elif line.startswith("#"):
                    continue
                elif line.startswith("Hex"):
                    header = line
                elif line.startswith("----"):
                    separator = line
                    fields = parse_header_and_separator(header, separator)
                else:
                    world = World(line, fields, self)
                    self.hex_to_world[world.hex_] = world

    def parse_xml_routes(self, data_dir: str):
        """Must be called after all Sectors and Worlds are otherwise built."""
        xml_path = os.path.join(data_dir, self.name + ".xml")
        tree = ET.parse(xml_path)
        root_element = tree.getroot()
        routes_element = root_element.find("Routes")
        if not routes_element:
            return
        for route_element in routes_element.findall("Route"):
            start_hex = route_element.attrib["Start"]
            end_hex = route_element.attrib["End"]
            start_offset_x = route_element.attrib.get("StartOffsetX", 0)
            start_offset_x = int(start_offset_x)
            start_offset_y = route_element.attrib.get("StartOffsetY", 0)
            start_offset_y = int(start_offset_y)
            end_offset_x = route_element.attrib.get("EndOffsetX", 0)
            end_offset_x = int(end_offset_x)
            end_offset_y = route_element.attrib.get("EndOffsetY", 0)
            end_offset_y = int(end_offset_y)
            start_sector = location_to_sector.get(
                (
                    self.location[0] + start_offset_x,
                    self.location[1] + start_offset_y,
                )
            )
            end_sector = location_to_sector.get(
                (
                    self.location[0] + end_offset_x,
                    self.location[1] + end_offset_y,
                )
            )
            if start_sector is not None and end_sector is not None:
                start_world = start_sector.hex_to_world[start_hex]
                end_world = end_sector.hex_to_world[end_hex]
                start_world.xboat_routes.add(end_world)
                end_world.xboat_routes.add(start_world)

    def populate_neighbors(self):
        for world in self.hex_to_world.values():
            world.populate_neighbors()

    @property
    def name(self):
        return self.names[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sector",
        "-s",
        action="append",
        dest="sector_names",
        required=True,
        help="sector in the format used by travellermap.com, e.g. 'Deneb'",
    )
    parser.add_argument(
        "--data-directory",
        "-d",
        action="store",
        help="directory for data files",
    )
    args = parser.parse_args()
    if args.data_directory:
        data_dir = args.data_directory
        tempdir = None
    else:
        tempdir = tempfile.mkdtemp(prefix="traderoutes.py")
        data_dir = tempdir
    download_sector_data(data_dir, args.sector_names)
    for sector_name in args.sector_names:
        sector = Sector(data_dir, sector_name)
    for sector in location_to_sector.values():
        sector.parse_xml_routes(data_dir)
        sector.populate_neighbors()

    if tempdir is not None:
        shutil.rmtree(tempdir)


if __name__ == "__main__":
    main()
