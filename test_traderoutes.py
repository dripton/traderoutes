# Run with pytest (or python3.8 -m pytest)

import os
import shutil
import stat
import tempfile

import pytest
import traderoutes as tr


@pytest.fixture(scope="session")
def tempdir():
    tempdir = tempfile.mkdtemp(prefix="test_traderoutes.py")
    yield tempdir
    shutil.rmtree(tempdir)


@pytest.fixture(scope="session")
def download(tempdir):
    # Fails if network or travellermap.com is down.
    sector_names = ["Deneb", "Spinward Marches"]
    tr.download_sector_data(tempdir, sector_names)


def test_download_sector_data(tempdir, download):
    sector_names = ["Deneb", "Spinward Marches"]
    all_filenames = []
    for sector_name in sector_names:
        all_filenames.append(sector_name)
        all_filenames.append(sector_name + ".xml")
    assert sorted(os.listdir(tempdir)) == all_filenames
    for filename in os.listdir(tempdir):
        path = os.path.join(tempdir, filename)
        stat_tuple = os.stat(path)
        assert stat_tuple[stat.ST_SIZE] > 0


def test_parse_header_and_separator():
    header = (
        "Hex  Name                 UWP       "
        "Remarks                                  {Ix}   (Ex)   "
        "[Cx]   N     B  Z PBG W  A    Stellar       "
    )
    separator = (
        "---- -------------------- --------- "
        "---------------------------------------- ------ ------- "
        "------ ----- -- - --- -- ---- --------------"
    )
    fields = tr.parse_header_and_separator(header, separator)
    assert len(fields) == 14
    assert fields["Hex"] == (0, 4)
    assert fields["Name"] == (5, 25)
    assert fields["UWP"] == (26, 35)
    assert fields["Remarks"] == (36, 76)
    assert fields["{Ix}"] == (77, 83)
    assert fields["(Ex)"] == (84, 91)
    assert fields["[Cx]"] == (92, 98)
    assert fields["N"] == (99, 104)
    assert fields["B"] == (105, 107)
    assert fields["Z"] == (108, 109)
    assert fields["PBG"] == (110, 113)
    assert fields["W"] == (114, 116)
    assert fields["A"] == (117, 121)
    assert fields["Stellar"] == (122, 136)


def test_same_allegiance():
    assert not tr.same_allegiance("CsIm", "CsIm")
    assert not tr.same_allegiance("CsZh", "CsZh")
    assert not tr.same_allegiance("CsIm", "CsZh")
    assert not tr.same_allegiance("NaHu", "NaHu")
    assert not tr.same_allegiance("NaXX", "NaXX")
    assert not tr.same_allegiance("NaHu", "NaXX")
    assert not tr.same_allegiance("DaCf", "ImDd")
    assert not tr.same_allegiance("ImDd", "ZhIN")
    assert tr.same_allegiance("DaCf", "DaCf")
    assert tr.same_allegiance("ImDd", "ImDd")
    assert tr.same_allegiance("SwCf", "SwCf")
    assert tr.same_allegiance("ZhIN", "ZhIN")


@pytest.fixture(scope="session")
def spin(tempdir, download):
    sector = tr.Sector(tempdir, "Spinward Marches")
    yield sector


def test_sector_spin(spin):
    sector = spin
    assert sector.name == "Spinward Marches"
    assert sector.names == ["Spinward Marches", "Tloql (zh)"]
    assert sector.abbreviation == "Spin"
    assert sector.location == (-4, -1)
    assert len(sector.subsector_letter_to_name) == 16
    assert sector.subsector_letter_to_name["A"] == "Cronor"
    assert sector.subsector_letter_to_name["P"] == "Trin's Veil"
    assert len(sector.allegiance_code_to_name) == 8
    assert (
        sector.allegiance_code_to_name["CsIm"]
        == "Client state, Third Imperium"
    )
    assert len(sector.hex_to_world) == 439
    assert sector.hex_to_world["0101"].name == "Zeycude"
    assert sector.hex_to_world["3236"].name == "Hazel"


@pytest.fixture(scope="session")
def dene(tempdir, download):
    sector = tr.Sector(tempdir, "Deneb")
    yield sector


def test_sector_dene(dene):
    sector = dene
    assert sector.name == "Deneb"
    assert sector.names == ["Deneb", "Nieklsdia (zh)"]
    assert sector.abbreviation == "Dene"
    assert sector.location == (-3, -1)
    assert len(sector.subsector_letter_to_name) == 16
    assert sector.subsector_letter_to_name["A"] == "Pretoria"
    assert sector.subsector_letter_to_name["P"] == "Vast Heavens"
    assert len(sector.allegiance_code_to_name) == 6
    assert (
        sector.allegiance_code_to_name["CsIm"]
        == "Client state, Third Imperium"
    )
    assert len(sector.hex_to_world) == 386
    assert sector.hex_to_world["0108"].name == "New Ramma"
    assert sector.hex_to_world["3031"].name == "Asharam"


def test_world_aramis(spin):
    sector = spin
    world = sector.hex_to_world["3110"]
    assert world.sector == spin
    assert world.name == "Aramis"
    assert world.hex_ == "3110"
    assert world.uwp == "A5A0556-B"
    assert world.trade_classifications == {"He", "Ni", "Cp"}
    assert world.importance == 2
    assert world.economic == "846+1"
    assert world.cultural == "474A"
    assert world.nobles == "BF"
    assert world.bases == {"N", "S"}
    assert world.zone == "G"
    assert world.pbg == "710"
    assert world.worlds == 9
    assert world.allegiance == "ImDd"
    assert world.stars == ["M2 V"]
    assert world.starport == "A"
    assert world.g_starport == "V"
    assert world.size == "5"
    assert world.atmosphere == "A"
    assert world.hydrosphere == "0"
    assert world.population == "5"
    assert world.government == "5"
    assert world.law_level == "6"
    assert world.tech_level == "B"
    assert world.g_tech_level == 9
    assert world.uwtn == 3.5
    assert world.wtn_port_modifier == 0.5
    assert world.wtn == 4.0


def test_world_regina(spin):
    sector = spin
    world = sector.hex_to_world["1910"]
    assert world.sector == spin
    assert world.name == "Regina"
    assert world.hex_ == "1910"
    assert world.uwp == "A788899-C"
    assert world.trade_classifications == {
        "Ri",
        "Pa",
        "Ph",
        "An",
        "Cp",
        "(Amindii)2",
        "Varg0",
        "Asla0",
        "Sa",
    }
    assert world.importance == 4
    assert world.economic == "D7E+5"
    assert world.cultural == "9C6D"
    assert world.nobles == "BcCeF"
    assert world.bases == {"N", "S"}
    assert world.zone == "G"
    assert world.pbg == "703"
    assert world.worlds == 8
    assert world.allegiance == "ImDd"
    assert world.stars == ["F7 V", "BD", "M3 V"]
    assert world.starport == "A"
    assert world.g_starport == "V"
    assert world.size == "7"
    assert world.atmosphere == "8"
    assert world.hydrosphere == "8"
    assert world.population == "8"
    assert world.government == "9"
    assert world.law_level == "9"
    assert world.tech_level == "C"
    assert world.g_tech_level == 10
    assert world.uwtn == 5.0
    assert world.wtn_port_modifier == 0.0
    assert world.wtn == 5.0


def test_abs_coords(spin, dene):
    aramis = spin.hex_to_world["3110"]
    ldd = spin.hex_to_world["3010"]
    natoko = spin.hex_to_world["3209"]
    reacher = spin.hex_to_world["3210"]
    vinorian = spin.hex_to_world["3111"]
    nutema = spin.hex_to_world["3112"]
    margesi = spin.hex_to_world["3212"]
    saarinen = dene.hex_to_world["0113"]
    regina = spin.hex_to_world["1910"]
    assert aramis.abs_coords == (-97, -30)
    assert ldd.abs_coords == (-98, -29.5)
    assert natoko.abs_coords == (-96, -30.5)
    assert reacher.abs_coords == (-96, -29.5)
    assert vinorian.abs_coords == (-97, -29)
    assert nutema.abs_coords == (-97, -28)
    assert margesi.abs_coords == (-96, -27.5)
    assert saarinen.abs_coords == (-95, -27)
    assert regina.abs_coords == (-109, -30)


def test_straight_line_distance(spin, dene):
    aramis = spin.hex_to_world["3110"]
    ldd = spin.hex_to_world["3010"]
    natoko = spin.hex_to_world["3209"]
    reacher = spin.hex_to_world["3210"]
    vinorian = spin.hex_to_world["3111"]
    nutema = spin.hex_to_world["3112"]
    margesi = spin.hex_to_world["3212"]
    saarinen = dene.hex_to_world["0113"]
    corfu = spin.hex_to_world["2602"]
    lablon = spin.hex_to_world["2701"]
    junidy = spin.hex_to_world["3202"]
    marz = dene.hex_to_world["0201"]
    regina = spin.hex_to_world["1910"]
    assert aramis.straight_line_distance(aramis) == 0
    assert aramis.straight_line_distance(ldd) == 1
    assert ldd.straight_line_distance(aramis) == 1
    assert aramis.straight_line_distance(natoko) == 1
    assert aramis.straight_line_distance(reacher) == 1
    assert natoko.straight_line_distance(reacher) == 1
    assert aramis.straight_line_distance(vinorian) == 1
    assert vinorian.straight_line_distance(nutema) == 1
    assert nutema.straight_line_distance(margesi) == 1
    assert margesi.straight_line_distance(saarinen) == 1
    assert ldd.straight_line_distance(natoko) == 2
    assert ldd.straight_line_distance(reacher) == 2
    assert ldd.straight_line_distance(nutema) == 2
    assert ldd.straight_line_distance(margesi) == 3
    assert ldd.straight_line_distance(saarinen) == 4
    assert aramis.straight_line_distance(corfu) == 10
    assert aramis.straight_line_distance(lablon) == 11
    assert aramis.straight_line_distance(junidy) == 8
    assert aramis.straight_line_distance(marz) == 10
    assert aramis.straight_line_distance(regina) == 12


def test_btn(spin, dene, neighbors):
    aramis = spin.hex_to_world["3110"]
    ldd = spin.hex_to_world["3010"]
    natoko = spin.hex_to_world["3209"]
    reacher = spin.hex_to_world["3210"]
    vinorian = spin.hex_to_world["3111"]
    nutema = spin.hex_to_world["3112"]
    margesi = spin.hex_to_world["3212"]
    saarinen = dene.hex_to_world["0113"]
    corfu = spin.hex_to_world["2602"]
    lablon = spin.hex_to_world["2701"]
    junidy = spin.hex_to_world["3202"]
    marz = dene.hex_to_world["0201"]
    regina = spin.hex_to_world["1910"]
    assert aramis.btn(ldd) == 8
    assert aramis.btn(natoko) == 6.5
    assert aramis.btn(reacher) == 7
    assert aramis.btn(vinorian) == 8
    assert aramis.btn(corfu) == 5.5
    assert aramis.btn(lablon) == 6
    assert aramis.btn(junidy) == 7.5
    assert aramis.btn(marz) == 7.5
    assert aramis.btn(regina) == 7
    assert ldd.btn(aramis) == 8
    assert ldd.btn(natoko) == 6
    assert ldd.btn(reacher) == 6.5
    assert ldd.btn(nutema) == 6
    assert ldd.btn(margesi) == 6
    assert ldd.btn(saarinen) == 5.5
    assert natoko.btn(reacher) == 5.5
    assert vinorian.btn(nutema) == 6.5
    assert nutema.btn(margesi) == 5.5
    assert margesi.btn(saarinen) == 5.5


def test_effective_passenger_btn(spin, dene, neighbors):
    aramis = spin.hex_to_world["3110"]
    ldd = spin.hex_to_world["3010"]
    natoko = spin.hex_to_world["3209"]
    reacher = spin.hex_to_world["3210"]
    vinorian = spin.hex_to_world["3111"]
    nutema = spin.hex_to_world["3112"]
    margesi = spin.hex_to_world["3212"]
    saarinen = dene.hex_to_world["0113"]
    corfu = spin.hex_to_world["2602"]
    lablon = spin.hex_to_world["2701"]
    junidy = spin.hex_to_world["3202"]
    marz = dene.hex_to_world["0201"]
    regina = spin.hex_to_world["1910"]
    assert aramis.effective_passenger_btn(ldd) == 8.5
    assert aramis.effective_passenger_btn(natoko) == 7
    assert aramis.effective_passenger_btn(reacher) == 7.5
    assert aramis.effective_passenger_btn(vinorian) == 8.5
    assert aramis.effective_passenger_btn(corfu) == 6
    assert aramis.effective_passenger_btn(lablon) == 6.5
    assert aramis.effective_passenger_btn(junidy) == 8
    assert aramis.effective_passenger_btn(marz) == 8
    assert aramis.effective_passenger_btn(regina) == 8.5
    assert ldd.effective_passenger_btn(aramis) == 8.5
    assert ldd.effective_passenger_btn(natoko) == 6
    assert ldd.effective_passenger_btn(reacher) == 6.5
    assert ldd.effective_passenger_btn(nutema) == 6
    assert ldd.effective_passenger_btn(margesi) == 6
    assert ldd.effective_passenger_btn(saarinen) == 5.5
    assert natoko.effective_passenger_btn(reacher) == 5.5
    assert vinorian.effective_passenger_btn(nutema) == 6.5
    assert nutema.effective_passenger_btn(margesi) == 5.5
    assert margesi.effective_passenger_btn(saarinen) == 5.5


@pytest.fixture(scope="session")
def xboat_routes(tempdir, spin, dene):
    spin.parse_xml_routes(tempdir)
    dene.parse_xml_routes(tempdir)


def test_xboat_routes(spin, dene, xboat_routes):
    aramis = spin.hex_to_world["3110"]
    ldd = spin.hex_to_world["3010"]
    natoko = spin.hex_to_world["3209"]
    reacher = spin.hex_to_world["3210"]
    vinorian = spin.hex_to_world["3111"]
    nutema = spin.hex_to_world["3112"]
    margesi = spin.hex_to_world["3212"]
    saarinen = dene.hex_to_world["0113"]
    corfu = spin.hex_to_world["2602"]
    lablon = spin.hex_to_world["2701"]
    junidy = spin.hex_to_world["3202"]
    marz = dene.hex_to_world["0201"]
    regina = spin.hex_to_world["1910"]
    celepina = spin.hex_to_world["2913"]
    teh = dene.hex_to_world["0208"]
    fennec = dene.hex_to_world["0311"]
    ash = dene.hex_to_world["0504"]
    roup = spin.hex_to_world["2007"]
    jenghe = spin.hex_to_world["1810"]
    dinomn = spin.hex_to_world["1912"]
    towers = spin.hex_to_world["3103"]
    assert aramis.xboat_routes == {ldd, natoko}
    assert ldd.xboat_routes == {aramis, celepina}
    assert natoko.xboat_routes == {aramis, teh}
    assert not reacher.xboat_routes
    assert not vinorian.xboat_routes
    assert not nutema.xboat_routes
    assert margesi.xboat_routes == {celepina, fennec}
    assert not saarinen.xboat_routes
    assert not corfu.xboat_routes
    assert not lablon.xboat_routes
    assert junidy.xboat_routes == {marz, towers}
    assert marz.xboat_routes == {junidy, ash}
    assert regina.xboat_routes == {roup, jenghe, dinomn}


@pytest.fixture(scope="session")
def neighbors(tempdir, spin, dene):
    spin.populate_neighbors()
    dene.populate_neighbors()


def test_neighbors(spin, dene, neighbors):
    aramis = spin.hex_to_world["3110"]
    ldd = spin.hex_to_world["3010"]
    natoko = spin.hex_to_world["3209"]
    reacher = spin.hex_to_world["3210"]
    vinorian = spin.hex_to_world["3111"]
    nutema = spin.hex_to_world["3112"]
    margesi = spin.hex_to_world["3212"]
    saarinen = dene.hex_to_world["0113"]
    corfu = spin.hex_to_world["2602"]
    lablon = spin.hex_to_world["2701"]
    junidy = spin.hex_to_world["3202"]
    marz = dene.hex_to_world["0201"]
    regina = spin.hex_to_world["1910"]
    celepina = spin.hex_to_world["2913"]
    teh = dene.hex_to_world["0208"]
    fennec = dene.hex_to_world["0311"]
    ash = dene.hex_to_world["0504"]
    roup = spin.hex_to_world["2007"]
    jenghe = spin.hex_to_world["1810"]
    dinomn = spin.hex_to_world["1912"]
    towers = spin.hex_to_world["3103"]
    pysadi = spin.hex_to_world["3008"]
    zila = spin.hex_to_world["2908"]
    lewis = spin.hex_to_world["3107"]
    patinir = spin.hex_to_world["3207"]
    henoz = spin.hex_to_world["2912"]
    valhalla = spin.hex_to_world["2811"]
    suvfoto = dene.hex_to_world["0211"]
    kretikaa = dene.hex_to_world["0209"]
    new_ramma = dene.hex_to_world["0108"]
    assert aramis.neighbors1 == {ldd, natoko, reacher, vinorian}
    assert aramis.neighbors2 == {nutema, pysadi}
    assert aramis.neighbors3 == {
        margesi,
        teh,
        zila,
        lewis,
        patinir,
        henoz,
        suvfoto,
        kretikaa,
        new_ramma,
        valhalla,
    }


def test_navigable_path(spin, dene, neighbors):
    aramis = spin.hex_to_world["3110"]
    ldd = spin.hex_to_world["3010"]
    natoko = spin.hex_to_world["3209"]
    reacher = spin.hex_to_world["3210"]
    vinorian = spin.hex_to_world["3111"]
    nutema = spin.hex_to_world["3112"]
    margesi = spin.hex_to_world["3212"]
    saarinen = dene.hex_to_world["0113"]
    corfu = spin.hex_to_world["2602"]
    lablon = spin.hex_to_world["2701"]
    junidy = spin.hex_to_world["3202"]
    marz = dene.hex_to_world["0201"]
    regina = spin.hex_to_world["1910"]
    celepina = spin.hex_to_world["2913"]
    teh = dene.hex_to_world["0208"]
    fennec = dene.hex_to_world["0311"]
    ash = dene.hex_to_world["0504"]
    roup = spin.hex_to_world["2007"]
    jenghe = spin.hex_to_world["1810"]
    dinomn = spin.hex_to_world["1912"]
    towers = spin.hex_to_world["3103"]
    pysadi = spin.hex_to_world["3008"]
    zila = spin.hex_to_world["2908"]
    lewis = spin.hex_to_world["3107"]
    patinir = spin.hex_to_world["3207"]
    henoz = spin.hex_to_world["2912"]
    valhalla = spin.hex_to_world["2811"]
    violante = spin.hex_to_world["2708"]
    focaline = spin.hex_to_world["2607"]
    moughas = spin.hex_to_world["2406"]
    keng = spin.hex_to_world["2405"]
    becks_world = spin.hex_to_world["2204"]
    yorbund = spin.hex_to_world["2303"]
    heya = spin.hex_to_world["2402"]
    suvfoto = dene.hex_to_world["0211"]
    kretikaa = dene.hex_to_world["0209"]
    new_ramma = dene.hex_to_world["0108"]
    reno = spin.hex_to_world["0102"]
    javan = dene.hex_to_world["2131"]
    assert aramis.navigable_path(ldd) == [ldd]
    assert aramis.navigable_path(corfu) == [
        pysadi,
        zila,
        violante,
        focaline,
        moughas,
        keng,
        becks_world,
        yorbund,
        heya,
        corfu,
    ]
    assert len(reno.navigable_path(javan)) == 33


def test_navigable_distance(spin, dene, neighbors):
    aramis = spin.hex_to_world["3110"]
    ldd = spin.hex_to_world["3010"]
    natoko = spin.hex_to_world["3209"]
    reacher = spin.hex_to_world["3210"]
    vinorian = spin.hex_to_world["3111"]
    nutema = spin.hex_to_world["3112"]
    margesi = spin.hex_to_world["3212"]
    saarinen = dene.hex_to_world["0113"]
    corfu = spin.hex_to_world["2602"]
    lablon = spin.hex_to_world["2701"]
    junidy = spin.hex_to_world["3202"]
    marz = dene.hex_to_world["0201"]
    regina = spin.hex_to_world["1910"]
    celepina = spin.hex_to_world["2913"]
    teh = dene.hex_to_world["0208"]
    fennec = dene.hex_to_world["0311"]
    ash = dene.hex_to_world["0504"]
    roup = spin.hex_to_world["2007"]
    jenghe = spin.hex_to_world["1810"]
    dinomn = spin.hex_to_world["1912"]
    towers = spin.hex_to_world["3103"]
    pysadi = spin.hex_to_world["3008"]
    zila = spin.hex_to_world["2908"]
    lewis = spin.hex_to_world["3107"]
    patinir = spin.hex_to_world["3207"]
    henoz = spin.hex_to_world["2912"]
    valhalla = spin.hex_to_world["2811"]
    suvfoto = dene.hex_to_world["0211"]
    kretikaa = dene.hex_to_world["0209"]
    new_ramma = dene.hex_to_world["0108"]
    reno = spin.hex_to_world["0102"]
    javan = dene.hex_to_world["2131"]
    assert aramis.navigable_distance(aramis) == 0
    assert aramis.navigable_distance(ldd) == 1
    assert aramis.navigable_distance(corfu) == 16
    assert reno.navigable_distance(javan) == 61
