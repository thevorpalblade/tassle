#!/usr/bin/python
# A module to simulate the motion of a point on the earth's surface through
# the galactic dark matter halo
#

import astropy
from astropy import coordinates as coord
from astropy.time import Time
from astropy import units as u
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

CASPEr_lat = 49.9916 # degrees north
CASPER_lon = 8.2353 # degrees east
CASPER_elevation = 130 # meters

def get_halo_vel(time=None):
    if time is None:
        time = Time.now()
    earth = coord.get_body_barycentric_posvel("earth", time)
    ediff = coord.CartesianDifferential(earth[1].xyz)
    earth_with_diff = earth[0].with_differentials(ediff)
    earth_coord = coord.SkyCoord(earth_with_diff)

    dm_vel = earth_coord.galactocentric.cartesian.differentials['s']

    CASPEr_loc = coord.EarthLocation(lon=CASPER_lon, lat=CASPEr_lat,
                                     height=CASPER_elevation)
    CASPEr_inst_loc = coord.SkyCoord(CASPEr_loc.get_itrs(time))


    return dm_vel

def convert_to_float128(sky_coord):
    cart_data = sky_coord.cartesian.xyz
    cart_data_f128 = coord.CartesianRepresentation(u.Quantity(cart_data,
                                                              dtype=np.float128))
    new_sc = coord.SkyCoord(cart_data_f128, frame=sky_coord.frame)
    return new_sc


def get_CASPEr_vect(time=None, lat=CASPEr_lat, lon=CASPER_lon,
                    elev=CASPER_elevation):
    # if we don't have a time array passed in, just use right now
    if time is None:
        time = Time.now()
    # location on earth of CASPEr
    CASPEr_loc = coord.EarthLocation(lon=lon, lat=lat,
                                     height=elev)
    # location in the solar system of CASPEr at a given moment
    CASPEr_inst_loc = coord.SkyCoord(CASPEr_loc.get_itrs(time))
    # recast as float128
    CASPEr_inst_loc = convert_to_float128(CASPEr_inst_loc)

    # location of the center of mass of the earth
    earth = coord.get_body_barycentric_posvel("earth", time)
    earth_coord = coord.SkyCoord(earth[0])
    earth_coord = convert_to_float128(earth_coord)
    # vector pointing between the center of the earth and CASPEr (should be in
    # the direction of H_0 in CASPEr)
    CASPEr_vec = (-earth_coord.galactocentric.cartesian
                  + CASPEr_inst_loc.galactocentric.cartesian)
    CASPEr_vec = CASPEr_vec/CASPEr_vec.norm()

    # now make orthogonal vectors
    CASPEr_north = coord.EarthLocation(lon=lon, lat=lat + 0.001,
                                       height=elev)
    CASPEr_east = coord.EarthLocation(lon=lon + 0.001, lat=lat,
                                       height=elev)

    CASPEr_north_inst = coord.SkyCoord(CASPEr_north.get_itrs(time))
    CASPEr_north_inst = convert_to_float128(CASPEr_north_inst)
    CASPEr_east_inst = coord.SkyCoord(CASPEr_east.get_itrs(time))
    CASPEr_east_inst = convert_to_float128(CASPEr_east_inst)

    north = (CASPEr_north_inst.galactocentric.cartesian
             - CASPEr_inst_loc.galactocentric.cartesian)
    north = north / north.norm()

    east = (CASPEr_east_inst.galactocentric.cartesian
            - CASPEr_inst_loc.galactocentric.cartesian)
    east = east / east.norm()


    return CASPEr_inst_loc, CASPEr_north_inst, CASPEr_east_inst, CASPEr_vec, north, east
    CASPEr_vec.dot(get_halo_vel(time))

def plot_dm_component(time=None):
    if time is None:
        time = Time.now()

    times = np.arange(1535018948.6150093, 1535018948.6150093+60*60*24*365*2, 60)
    t = Time(times, format='unix')

    halo_vel = get_halo_vel(t)
    casper_vect = get_CASPEr_vect(t)

    component = casper_vect.cross(halo_vel).norm()
    plt.plot(component)
