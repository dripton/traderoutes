#!/usr/bin/env python3.8

"""Calculate trade routes for GURPS Traveller: Far Trader"""

# Needed for circular class references in mypy.
from __future__ import annotations

import argparse
import math
import os
import shutil
import tempfile
from typing import Any, Dict, List, Set, Tuple
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET


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


def download_sector_data(data_dir: str, sector_names: List[str]) -> None:
    for sector in sector_names:
        data_path = os.path.join(data_dir, sector)
        sector_xml_filename = sector + ".xml"
        metadata_path = os.path.join(data_dir, sector_xml_filename)
        escaped_sector = urllib.parse.quote(sector)
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
    xboat_routes: Set[Any]  # Should be Set[World]

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

    def __str__(self):
        return "World: " + self.name

    def __repr__(self):
        return "World: " + self.name

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
        tech_level_int = int(self.tech_level, 16)
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

    def abs_coords(self) -> Tuple[float, float]:
        hex_ = self.hex_
        location = self.sector.location
        x = int(hex_[:2]) + 32 * location[0]
        y = int(hex_[2:]) + 40 * location[1] + 0.5 * (x & 1 == 0)
        return (x, y)

    def straight_line_distance(self, other: World) -> int:
        """Return the shortest distance in hexes between the worlds"""
        (x1, y1) = self.abs_coords()
        (x2, y2) = other.abs_coords()
        xdelta = abs(x2 - x1)
        ydelta = max(0, abs(y2 - y1) - xdelta / 2)
        return math.floor(xdelta + ydelta)

    def distance_modifier(self, other: World) -> float:
        # TODO This should be navigable distance
        # jump-4 along Xboat routes, jump-2 otherwise
        distance = self.straight_line_distance(other)
        if distance <= 1:
            return 0.0
        elif distance <= 2:
            return 0.5
        elif distance <= 5:
            return 1.0
        elif distance <= 9:
            return 1.5
        elif distance <= 19:
            return 2.0
        elif distance <= 29:
            return 2.5
        elif distance <= 59:
            return 3.0
        elif distance <= 99:
            return 3.5
        elif distance <= 199:
            return 4.0
        elif distance <= 299:
            return 4.5
        elif distance <= 599:
            return 5.0
        elif distance <= 999:
            return 5.5
        else:
            return 6.0

    def btn(self, other: World) -> float:
        btn = (
            self.wtn
            + other.wtn
            + self.wtcm(other)
            - self.distance_modifier(other)
        )
        min_wtn = min(self.wtn, other.wtn)
        return min(btn, min_wtn + 5)

    def effective_passenger_btn(self, other: World) -> float:
        result = self.btn(other)
        for world in [self, other]:
            if "Ri" in world.trade_classifications:
                result += 0.5
            if "Cp" in world.trade_classifications:
                result += 0.5
            if "Cs" in world.trade_classifications:
                result += 1.0
        return result


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

    def __str__(self):
        return "Sector: " + self.name

    def __repr__(self):
        return "Sector: " + self.name

    def parse_xml_metadata(self, data_dir: str, sector_name: str):
        xml_path = os.path.join(data_dir, sector_name + ".xml")
        tree = ET.parse(xml_path)
        root_element = tree.getroot()
        self.abbreviation = root_element.attrib["Abbreviation"]
        x = 9999
        x_element = root_element.find("X")
        if x_element is not None:
            x_text = x_element.text
            if x_text:
                x = int(x_text)
        y = 9999
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
        # Set this last after the sector is as fully built as possible.
        location_to_sector[self.location] = self

    def parse_column_data(self, data_dir: str, sector_name: str):
        data_path = os.path.join(data_dir, sector_name)
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

    if tempdir is not None:
        shutil.rmtree(tempdir)


if __name__ == "__main__":
    main()
