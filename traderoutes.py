#!/usr/bin/env python3.8

"""Calculate trade routes for GURPS Traveller: Far Trader"""

# Needed for circular class references in mypy.
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import defaultdict
from math import floor, pi
import os
import random
import shutil
from sys import maxsize
import tempfile
from typing import Dict, List, Optional, Set, Tuple
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import cairo
import numpy
from scipy.sparse.csgraph import floyd_warshall


SQRT3 = 3.0 ** 0.5


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

# Global so we only have to compute these once each
navigable_dist_info2 = None  # type Optional[NavigableDistanceInfo]
navigable_dist_info3 = None  # type Optional[NavigableDistanceInfo]


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
    """Must be run after navigable distances are built."""
    wtn_worlds = []
    for world in sorted_worlds:
        wtn_worlds.append((world.wtn, world))
    wtn_worlds.sort()
    wtn_worlds.reverse()
    return wtn_worlds


class NavigableDistanceInfo:
    navigable_dist: Dict[Tuple[World, World], int]
    predecessors: numpy.ndarray

    def __init__(self, navigable_dist, predecessors):
        self.navigable_dist = navigable_dist
        self.predecessors = predecessors


def populate_navigable_distances(max_jump: int) -> NavigableDistanceInfo:
    """Find minimum distances between all worlds using the Floyd-Warshall
    algorithm.

    Only use jumps of up to max_jump hexes, except along xboat routes.

    Must be run after all neighbors are built.

    TODO Skip worlds with no gas giants, water, or class D+ starport.
    TODO Skip red zone worlds with no gas giants.
    """
    global sorted_worlds
    sorted_worlds = sorted(abs_coords_to_world.values())
    index_to_world = {}
    for ii, world in enumerate(sorted_worlds):
        world.index = ii
        index_to_world[ii] = world
    nd = numpy.full((len(sorted_worlds), len(sorted_worlds)), maxsize)
    for ii, world in enumerate(sorted_worlds):
        if max_jump >= 3:
            for neighbor3 in world.neighbors3:
                nd[ii][neighbor3.index] = 3
        if max_jump >= 2:
            for neighbor2 in world.neighbors2:
                nd[ii][neighbor2.index] = 2
        if max_jump >= 1:
            for neighbor1 in world.neighbors1:
                nd[ii][neighbor1.index] = 1
        for neighbor in world.xboat_routes:
            nd[ii][neighbor.index] = world.straight_line_distance(neighbor)
        nd[ii][ii] = 0

    dist_matrix, predecessors = floyd_warshall(
        nd, directed=False, return_predecessors=True
    )
    navigable_dist = {}
    for yy, row in enumerate(dist_matrix):
        world1 = index_to_world[yy]
        for xx, dist in enumerate(row):
            world2 = index_to_world[xx]
            navigable_dist[world1, world2] = dist
    return NavigableDistanceInfo(navigable_dist, predecessors)


def populate_trade_routes() -> None:
    """Fill in major_routes, main_routes, intermediate_routes, minor_routes,
    and feeder_routes for all Worlds.

    This must be called after all Sectors and Worlds are mostly built.
    The rules say: main: 10+  feeder: 9-9.5  minor: 8-8.5
    The wiki says: blue major 12, cyan main 11, green intermediate 10,
                   yellow feeder 9, red minor 8, no line 1-7
    The wiki version is more fun so we'll use that.
    """
    # TODO Track endpoint traffic and transient traffic
    # TODO If there are two routes pick the one with more traffic.
    # TODO Try to avoid Red and Amber Zones
    # TODO Try to avoid poor starports
    # TODO Try to avoid worlds of different allegiance
    wtn_worlds = worlds_by_wtn()

    # Add initial endpoint-only routes to both endpoints
    for ii, (wtn1, world1) in enumerate(wtn_worlds):
        for jj in range(ii + 1, len(wtn_worlds)):
            (wtn2, world2) = wtn_worlds[jj]
            btn = world1.btn(world2)
            if btn >= 12:
                world1.major_routes.add(world2)
                world2.major_routes.add(world1)
            elif btn >= 11:
                world1.main_routes.add(world2)
                world2.main_routes.add(world1)
            elif btn >= 10:
                world1.intermediate_routes.add(world2)
                world2.intermediate_routes.add(world1)
            elif btn >= 9:
                world1.feeder_routes.add(world2)
                world2.feeder_routes.add(world1)
            elif btn >= 8:
                world1.minor_routes.add(world2)
                world2.minor_routes.add(world1)

    # Find all the route paths
    dd = defaultdict
    major_route_paths = dd(int)  # type: Dict[Tuple[World, World], int]
    main_route_paths = dd(int)  # type: Dict[Tuple[World, World], int]
    intermediate_route_paths = dd(int)  # type: Dict[Tuple[World, World], int]
    feeder_route_paths = dd(int)  # type: Dict[Tuple[World, World], int]
    minor_route_paths = dd(int)  # type: Dict[Tuple[World, World], int]

    def find_route_paths(
        route_paths: Dict[Tuple[World, World], int],
        routes: Set[World],
        max_jump: int,
    ) -> None:
        for world2 in routes:
            possible_path = world1.navigable_path(world2, 2)
            possible_path3 = None
            if max_jump == 3:
                possible_path3 = world1.navigable_path(world2, 3)
            if possible_path is None or (
                possible_path3 is not None
                and len(possible_path3) < len(possible_path)
            ):
                possible_path = possible_path3
            if possible_path is not None:
                path = [world1] + possible_path
                if len(path) >= 2:
                    for ii in range(len(path) - 1):
                        first = path[ii]
                        second = path[ii + 1]
                        world_list = sorted(
                            [first, second]
                        )  # type: List[World]
                        world_tuple = tuple(world_list)
                        route_paths[world_tuple] += 1  # type: ignore

    for unused, world1 in wtn_worlds:
        find_route_paths(major_route_paths, world1.major_routes, 3)
        find_route_paths(main_route_paths, world1.main_routes, 3)
        find_route_paths(
            intermediate_route_paths, world1.intermediate_routes, 3
        )
        find_route_paths(feeder_route_paths, world1.feeder_routes, 3)
        find_route_paths(minor_route_paths, world1.minor_routes, 2)

    def promote_routes(smaller_route_paths, bigger_route_paths):
        for (world1, world2), count in smaller_route_paths.items():
            if count >= 3:
                bigger_route_paths[(world1, world2)] += 1
                smaller_route_paths[(world1, world2)] = 0
        # Clear out elements with zero counts
        new_smaller_route_paths = {}  # type: Dict[Tuple[World, World], int]
        for (world1, world2), count in smaller_route_paths.items():
            if count > 0:
                new_smaller_route_paths[(world1, world2)] = count
        return new_smaller_route_paths, bigger_route_paths

    minor_route_paths, feeder_route_paths = promote_routes(
        minor_route_paths, feeder_route_paths
    )
    feeder_route_paths, intermediate_route_paths = promote_routes(
        feeder_route_paths, intermediate_route_paths
    )
    intermediate_route_paths, main_route_paths = promote_routes(
        intermediate_route_paths, main_route_paths
    )
    main_route_paths, major_route_paths = promote_routes(
        main_route_paths, major_route_paths
    )

    # Clear out the initial routes and fill in the full versions.
    for unused, world1 in wtn_worlds:
        world1.major_routes = set()
        world1.main_routes = set()
        world1.intermediate_routes = set()
        world1.feeder_routes = set()
        world1.minor_routes = set()

    # Keep the largest route for each pair of worlds.
    for (world1, world2) in minor_route_paths:
        world1.minor_routes.add(world2)
        world2.minor_routes.add(world1)
    for (world1, world2) in feeder_route_paths:
        world1.feeder_routes.add(world2)
        world2.feeder_routes.add(world1)
        world1.minor_routes.discard(world2)
        world2.minor_routes.discard(world1)
    for (world1, world2) in intermediate_route_paths:
        world1.intermediate_routes.add(world2)
        world2.intermediate_routes.add(world1)
        world1.feeder_routes.discard(world2)
        world2.feeder_routes.discard(world1)
        world1.minor_routes.discard(world2)
        world2.minor_routes.discard(world1)
    for (world1, world2) in main_route_paths:
        world1.main_routes.add(world2)
        world2.main_routes.add(world1)
        world1.intermediate_routes.discard(world2)
        world2.intermediate_routes.discard(world1)
        world1.feeder_routes.discard(world2)
        world2.feeder_routes.discard(world1)
        world1.minor_routes.discard(world2)
        world2.minor_routes.discard(world1)
    for (world1, world2) in major_route_paths:
        world1.major_routes.add(world2)
        world2.major_routes.add(world1)
        world1.main_routes.discard(world2)
        world2.main_routes.discard(world1)
        world1.intermediate_routes.discard(world2)
        world2.intermediate_routes.discard(world1)
        world1.feeder_routes.discard(world2)
        world2.feeder_routes.discard(world1)
        world1.minor_routes.discard(world2)
        world2.minor_routes.discard(world1)


def generate_pdfs(output_dir: str) -> None:
    """Generate PDF output for each sector.

    Sectors are 32 hexes wide by 40 hexes tall.  Subsectors are 8x10.
    Even-numbered hexes are shifted one-half hex down.
    Hexes are flat on top and bottom.
    The top left hex of a sector is 0101, with 0102 below it.
    """
    for sector in location_to_sector.values():
        generate_pdf(sector, output_dir)


def generate_pdf(sector: Sector, output_dir: str) -> None:
    """Generate PDF output for sector.

    Sectors are 32 hexes wide by 40 hexes tall.
    Even-numbered hexes are shifted one-half hex down.
    Hexes are flat on top and bottom.
    The top left hex of a sector is 0101, with 0102 below it.
    The 16 subsectors within the sector are each 8x10.

    TODO Subsector names and borders
    TODO Adjacent sector names
    TODO Allegiance borders
    TODO Bases
    TODO Research stations
    """

    def draw_route(worlds, line_width, rgba):
        for world2 in worlds:
            x2, y2 = world2.abs_coords
            delta_x = x2 - x1
            delta_y = y2 - y1
            cx2 = cx + delta_x * 3 * scale
            cy2 = cy + delta_y * 2 * SQRT3 * scale
            center2 = (cx2 + 2 * scale, cy2 + SQRT3 * scale)
            ctx.set_line_width(line_width)
            ctx.set_source_rgba(*rgba)
            ctx.move_to(*center)
            ctx.line_to(*center2)
            ctx.stroke()

    width = 25000  # TODO maybe have size options
    height = 35000
    output_filename = f"{sector.name}.pdf"
    output_path = os.path.join(output_dir, output_filename)
    scale = 15
    sector_hex_width = 32
    sector_hex_height = 40
    with cairo.PDFSurface(output_path, width, height) as surface:
        ctx = cairo.Context(surface)
        ctx.scale(scale, scale)
        ctx.set_source_rgba(0.0, 0.0, 0.0, 1.0)  # black background
        ctx.rectangle(0, 0, width, height)
        ctx.fill()
        normal_font_face = cairo.ToyFontFace("Sans")
        bold_font_face = cairo.ToyFontFace(
            "Sans", cairo.FontSlant.NORMAL, cairo.FontWeight.BOLD
        )

        # sector name
        ctx.set_font_size(3 * scale)
        ctx.set_font_face(bold_font_face)
        ctx.set_source_rgba(1.0, 1.0, 1.0, 1.0)  # white
        text = sector.name
        extents = ctx.text_extents(text)
        ctx.move_to(width / scale / 2 - extents.width / 2, 3 * scale)
        ctx.show_text(text)

        # first pass through hexes; draw hexsides
        for x in range(1, sector_hex_width + 1):
            for y in range(1, sector_hex_height + 1):
                hex_ = f"{x:02}{y:02}"
                cx = (x + 1) * 3 * scale  # leftmost point
                cy = (y * 2 + ((x - 1) & 1)) * SQRT3 * scale  # topmost point
                vertexes = []  # start at top left and go clockwise
                vertexes.append((cx + scale, cy))
                vertexes.append((cx + 3 * scale, cy))
                vertexes.append((cx + 4 * scale, cy + SQRT3 * scale))
                vertexes.append((cx + 3 * scale, cy + 2 * SQRT3 * scale))
                vertexes.append((cx + scale, cy + 2 * SQRT3 * scale))
                vertexes.append((cx, cy + SQRT3 * scale))
                center = (cx + 2 * scale, cy + SQRT3 * scale)

                # hexsides
                ctx.set_line_width(0.03 * scale)
                ctx.move_to(*vertexes[0])
                ctx.set_source_rgba(1.0, 1.0, 1.0, 1.0)  # white
                for ii in [1, 2, 3, 4, 5, 0]:
                    ctx.line_to(*vertexes[ii])
                ctx.stroke()

        # second pass through hexes; draw Xboat routes
        for x in range(1, sector_hex_width + 1):
            for y in range(1, sector_hex_height + 1):
                hex_ = f"{x:02}{y:02}"
                cx = (x + 1) * 3 * scale  # leftmost point
                cy = (y * 2 + ((x - 1) & 1)) * SQRT3 * scale  # topmost point
                vertexes = []  # start at top left and go clockwise
                vertexes.append((cx + scale, cy))
                vertexes.append((cx + 3 * scale, cy))
                vertexes.append((cx + 4 * scale, cy + SQRT3 * scale))
                vertexes.append((cx + 3 * scale, cy + 2 * SQRT3 * scale))
                vertexes.append((cx + scale, cy + 2 * SQRT3 * scale))
                vertexes.append((cx, cy + SQRT3 * scale))
                center = (cx + 2 * scale, cy + SQRT3 * scale)

                world = sector.hex_to_world.get(hex_)
                if world:
                    x1, y1 = world.abs_coords

                    # Xboat routes
                    draw_route(world.xboat_routes, 0.3 * scale, (0, 0, 1, 1))

        # third pass through hexes; draw trade routes
        for x in range(1, sector_hex_width + 1):
            for y in range(1, sector_hex_height + 1):
                hex_ = f"{x:02}{y:02}"
                cx = (x + 1) * 3 * scale  # leftmost point
                cy = (y * 2 + ((x - 1) & 1)) * SQRT3 * scale  # topmost point
                vertexes = []  # start at top left and go clockwise
                vertexes.append((cx + scale, cy))
                vertexes.append((cx + 3 * scale, cy))
                vertexes.append((cx + 4 * scale, cy + SQRT3 * scale))
                vertexes.append((cx + 3 * scale, cy + 2 * SQRT3 * scale))
                vertexes.append((cx + scale, cy + 2 * SQRT3 * scale))
                vertexes.append((cx, cy + SQRT3 * scale))
                center = (cx + 2 * scale, cy + SQRT3 * scale)

                world = sector.hex_to_world.get(hex_)
                if world:
                    x1, y1 = world.abs_coords

                    # Trade routes
                    draw_route(
                        world.major_routes, 0.05 * scale, (0, 0, 1, 0.5)
                    )
                    draw_route(
                        world.main_routes, 0.05 * scale, (0, 0.8, 0.8, 0.5)
                    )
                    draw_route(
                        world.intermediate_routes, 0.05 * scale, (0, 1, 0, 0.5)
                    )
                    draw_route(
                        world.feeder_routes, 0.05 * scale, (1, 1, 0, 0.5)
                    )
                    draw_route(
                        world.minor_routes, 0.05 * scale, (1, 0, 0, 0.5)
                    )

        # fourth pass through hexes
        # draw world, gas giants, text
        for x in range(1, sector_hex_width + 1):
            for y in range(1, sector_hex_height + 1):
                hex_ = f"{x:02}{y:02}"
                cx = (x + 1) * 3 * scale  # leftmost point
                cy = (y * 2 + ((x - 1) & 1)) * SQRT3 * scale  # topmost point
                vertexes = []  # start at top left and go clockwise
                vertexes.append((cx + scale, cy))
                vertexes.append((cx + 3 * scale, cy))
                vertexes.append((cx + 4 * scale, cy + SQRT3 * scale))
                vertexes.append((cx + 3 * scale, cy + 2 * SQRT3 * scale))
                vertexes.append((cx + scale, cy + 2 * SQRT3 * scale))
                vertexes.append((cx, cy + SQRT3 * scale))
                center = (cx + 2 * scale, cy + SQRT3 * scale)

                world = sector.hex_to_world.get(hex_)
                if world:
                    x1, y1 = world.abs_coords

                    # UWP
                    ctx.set_font_size(0.35 * scale)
                    ctx.set_font_face(normal_font_face)
                    ctx.set_source_rgba(1.0, 1.0, 1.0, 1.0)  # white
                    text = world.uwp
                    extents = ctx.text_extents(text)
                    ctx.move_to(
                        cx + 2 * scale - extents.width / 2,
                        cy + SQRT3 * scale * 1.5,
                    )
                    ctx.show_text(text)

                    # World name
                    # All-caps for high population
                    if world.population.isalpha() or world.population == "9":
                        text = world.name.upper()
                    else:
                        text = world.name
                    ctx.set_font_size(0.4 * scale)
                    ctx.set_font_face(bold_font_face)
                    extents = ctx.text_extents(text)
                    # Red if a sector or subsector capital
                    if (
                        "Cp" in world.trade_classifications
                        or "Cs" in world.trade_classifications
                    ):
                        ctx.set_source_rgba(1.0, 0.0, 0.0, 1.0)  # red
                    else:
                        ctx.set_source_rgba(1.0, 1.0, 1.0, 1.0)  # white
                    ctx.move_to(
                        cx + 2 * scale - extents.width / 2,
                        cy + SQRT3 * scale * 1.8,
                    )
                    ctx.show_text(text)

                    # World circle
                    if world.size == "0":
                        # Asteroid belt
                        rgba = (1.0, 1.0, 1.0, 1.0)  # white
                        ctx.set_source_rgba(*rgba)
                        num_asteroids = random.randrange(5, 20)
                        for unused in range(num_asteroids):
                            x_pos = (
                                center[0]
                                - 0.25 * scale
                                + random.random() * 0.5 * scale
                            )
                            y_pos = (
                                center[1]
                                - 0.25 * scale
                                + random.random() * 0.5 * scale
                            )
                            ctx.new_sub_path()
                            ctx.arc(
                                x_pos,
                                y_pos,
                                random.random() * 0.04 * scale,
                                0,
                                2 * pi,
                            )
                            ctx.stroke_preserve()
                            ctx.fill()
                    else:
                        if (
                            "Ri" in world.trade_classifications
                            and "Ag" in world.trade_classifications
                        ):
                            rgba = (1.0, 1.0, 0.0, 1.0)  # yellow
                            fill_rgba = rgba
                        elif "Ri" in world.trade_classifications:
                            rgba = (0.5, 0.0, 0.5, 1.0)  # purple
                            fill_rgba = rgba
                        elif "Ag" in world.trade_classifications:
                            rgba = (0.0, 1.0, 0.0, 1.0)  # green
                            fill_rgba = rgba
                        elif "In" in world.trade_classifications:
                            rgba = (0.5, 0.5, 0.5, 1.0)  # green
                            fill_rgba = rgba
                        elif world.atmosphere in {"B", "C"}:
                            rgba = (1.0, 0.65, 0.0, 1.0)  # orange
                            fill_rgba = rgba
                        elif world.atmosphere == "0":
                            rgba = (1.0, 1.0, 1.0, 1.0)  # white
                            fill_rgba = (0.0, 0.0, 0.0, 1.0)  # black
                        elif world.hydrosphere != "0":
                            rgba = (0.0, 0.0, 1.0, 1.0)  # blue
                            fill_rgba = rgba
                        else:
                            rgba = (1.0, 1.0, 1.0, 1.0)  # white
                            fill_rgba = rgba
                        ctx.set_source_rgba(*rgba)
                        ctx.new_sub_path()
                        ctx.arc(center[0], center[1], 0.3 * scale, 0, 2 * pi)
                        ctx.set_line_width(0.03 * scale)
                        ctx.stroke_preserve()
                        if fill_rgba != rgba:
                            ctx.set_source_rgba(*fill_rgba)
                        ctx.fill()

                    # gas giant
                    if world.gas_giants != "0":
                        rgba = (1.0, 1.0, 1.0, 1.0)  # white
                        ctx.set_source_rgba(*rgba)
                        ctx.new_sub_path()
                        ctx.arc(
                            center[0] + 0.8 * scale,
                            center[1] - 0.8 * scale,
                            0.1 * scale,
                            0.0,
                            2 * pi,
                        )
                        ctx.set_line_width(0.03 * scale)
                        ctx.stroke_preserve()
                        ctx.fill()

                    # red and amber zones
                    if world.zone in {"R", "A"}:
                        if world.zone == "R":
                            rgba = (1.0, 0.0, 0.0, 1.0)  # red
                        else:
                            rgba = (1.0, 1.0, 0.0, 1.0)  # yellow
                        ctx.set_source_rgba(*rgba)
                        ctx.new_sub_path()
                        ctx.arc(
                            center[0],
                            center[1],
                            1.5 * scale,
                            0.7 * pi,
                            2.3 * pi,
                        )
                        ctx.set_line_width(0.03 * scale)
                        ctx.stroke()

                # hex label
                text = hex_
                ctx.set_font_size(0.35 * scale)
                ctx.set_font_face(normal_font_face)
                extents = ctx.text_extents(text)

                ctx.set_source_rgba(1.0, 1.0, 1.0, 0.6)  # white
                ctx.move_to(
                    cx + 2 * scale - extents.width / 2,
                    cy + SQRT3 * scale * 0.3,
                )
                ctx.show_text(text)


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
    major_routes: Set[World]
    main_routes: Set[World]
    intermediate_routes: Set[World]
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
        self.major_routes = set()
        self.main_routes = set()
        self.intermediate_routes = set()
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
                self.nobles = value.strip(" -")
            elif field == "B":
                for ch in value.strip(" -"):
                    self.bases.add(ch)
            elif field == "Z":
                self.zone = value.strip(" -")
                if not self.zone:
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
    def population_multiplier(self) -> str:
        return self.pbg[0]

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
    def belts(self) -> str:
        return self.pbg[1]

    @property
    def gas_giants(self) -> str:
        return self.pbg[2]

    @property
    def can_refuel(self) -> bool:
        return (self.gas_giants != "0") or (
            self.zone != "R"
            and (self.starport not in {"E", "X"} or self.hydrosphere != "0")
        )

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
        return floor(xdelta + ydelta)

    def navigable_distance(self, other: World, max_jump: int) -> Optional[int]:
        """Return the length of the shortest navigable path from self to other.

        If it's not reachable, return None.
        This can only be called after populate_navigable_distances() runs.
        """
        if max_jump == 2 and navigable_dist_info2 is not None:
            navigable_dist = navigable_dist_info2.navigable_dist
        elif max_jump == 3 and navigable_dist_info3 is not None:
            navigable_dist = navigable_dist_info3.navigable_dist
        dist = navigable_dist[self, other]
        if dist == maxsize:
            return None
        return dist

    def navigable_path(
        self, other: World, max_jump: int
    ) -> Optional[List[World]]:
        """Return the shortest navigable path from self to other.

        If it's not reachable, return None.
        The path should include other but not self.
        This uses jump-4 only along Xboat routes, and jump-2 otherwise.
        This can only be called after populate_navigable_distances() runs.
        """
        if self == other:
            return []
        if self.navigable_distance(other, max_jump) is None:
            return None
        if max_jump == 2 and navigable_dist_info2 is not None:
            predecessors = navigable_dist_info2.predecessors
        elif max_jump == 3 and navigable_dist_info3 is not None:
            predecessors = navigable_dist_info3.predecessors
        world2 = self
        path = []
        while world2 != other:
            index = predecessors[other.index][world2.index]
            world2 = sorted_worlds[index]
            path.append(world2)
        return path

    def distance_modifier(self, other: World) -> float:
        distance = self.navigable_distance(other, 2)
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
        """Must be called after all Sectors and Worlds are otherwise built."""
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
    parser.add_argument(
        "--output-directory",
        "-o",
        action="store",
        help="directory for output files",
        default="/var/tmp",
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

    global navigable_dist_info2
    navigable_dist_info2 = populate_navigable_distances(2)
    global navigable_dist_info3
    navigable_dist_info3 = populate_navigable_distances(3)
    populate_trade_routes()
    generate_pdfs(args.output_directory)

    if tempdir is not None:
        shutil.rmtree(tempdir)


if __name__ == "__main__":
    main()
