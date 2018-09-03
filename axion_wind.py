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

def get_halo_vel(time=None, loc=None):
    """
    This is a function to compute the average velocity and direction of
    the earth within the galactic dark matter halo. The reason is to compute
    the velocity and the direction of the axion wind at the CASPEr detector
    (although this will be the average wind, not the instantaneous wind which
    will vary from one coherence time to another).

    It returns a CartesianRepresentation of CASPEr's velocity in the
    galactocentric frame.

    It accepts optional inputs time and loc.

    time should be a astropy Time object (possibly specifying an array of
    times, in which case the Cartesian representation returned will contain
    an array of locations as well). If not specified, the current time is used.

    loc should be a tuple like (latitude, longitude, elevation). If left
    blank, the location of CASPEr in Mainz is used.
    """
    # set up default time and location
    if time is None:
        time = Time.now()

    if loc is None:
        lat = CASPEr_lat
        lon = CASPER_lon
        elev = CASPER_elevation
    else:
        lat, lon, elev = loc
    # instantiate the location of CASPEr on the earth
    CASPEr_loc = coord.EarthLocation(lon=lon, lat=lat, height=elev)
    # make a sky_coord with CASPEr's location and velcity in GCRS coordinates
    CASPEr_inst_loc = coord.SkyCoord(CASPEr_loc.get_gcrs(time))
    # transform to the galactocentric frame and take the velocity
    return CASPEr_inst_loc.galactocentric.cartesian.differentials['s']

def convert_to_float128(sky_coord):
    """
    A utility function to convert the dtype of a skycoord to float128.
    It's not the best function, and strips some information from the skycoord
    (such as any differentials). It would be better to modify the data in-place
    the SkyCoord.data method, but I haven't figured out the details, and it
    probably isn't important.
    """
    cart_data = sky_coord.cartesian.xyz
    cart_data_f128 = coord.CartesianRepresentation(u.Quantity(cart_data,
                                                              dtype=np.float128))
    new_sc = coord.SkyCoord(cart_data_f128, frame=sky_coord.frame)
    return new_sc


def get_CASPEr_vect(time=None, lat=CASPEr_lat, lon=CASPER_lon,
                    elev=CASPER_elevation):
    """
    This function does the heavy lifting of the whole operation.

    It computes the unit vectors in the north, east, and up directions at
    the specified time and location, but in the galactocentric frame.

    time should be an astropy.time.Time object. If it specifies multiple times,
    multiple sets of unit vectors are returned.
    """
    # if we don't have a time array passed in, just use right now
    if time is None:
        time = Time.now()
    # location on earth of CASPEr
    CASPEr_loc = coord.EarthLocation(lon=lon, lat=lat,
                                     height=elev)
    # location in the solar system of CASPEr at a given moment in GCRS coords.
    CASPEr_inst_loc = coord.SkyCoord(CASPEr_loc.get_gcrs(time))
    # recast as float128
    CASPEr_inst_loc = convert_to_float128(CASPEr_inst_loc)
    # get the spherical representation of the GCRS coordinates
    CASPEr_sph = CASPEr_inst_loc.represent_as(coord.SphericalRepresentation)
    # set up differentials in the north, east, and up directions
    dnorth = coord.SphericalDifferential(d_lon=0.*u.arcsec/u.s,
                                         d_lat=1.*u.arcsec/u.s,
                                         d_distance=0.*u.km/u.s)
    deast = coord.SphericalDifferential(d_lon=1.*u.arcsec/u.s,
                                        d_lat=0.*u.arcsec/u.s,
                                        d_distance=0.*u.km/u.s)
    dup = coord.SphericalDifferential(d_lon=0.*u.arcsec/u.s,
                                      d_lat=0.*u.arcsec/u.s,
                                      d_distance=1.*u.km/u.s)
    # turn those differentials into displacements by multiplying in a large
    # time
    delta_north = dnorth * 1e15 * u.yr
    delta_east = deast * 1e15 * u.yr
    delta_up = dup * 1e15 * u.yr
    # turn those displacements into locations north, east, and up of the
    # experiment.
    casper_north  = coord.SkyCoord(CASPEr_sph + delta_north)
    casper_east = coord.SkyCoord(CASPEr_sph + delta_east)
    casper_up = coord.SkyCoord(CASPEr_sph + delta_up)
    # transform those locations into galactocentric coordinates and subtract
    # off casper's location to get vectors pointing north, east, and up, but
    # transformed to the galactocentric frame
    galactocentric_up = (casper_up.galactocentric.cartesian
                         - CASPEr_inst_loc.galactocentric.cartesian)
    galactocentric_north = (casper_north.galactocentric.cartesian
                            - CASPEr_inst_loc.galactocentric.cartesian)
    galactocentric_east = (casper_east.galactocentric.cartesian
                           - CASPEr_inst_loc.galactocentric.cartesian)
    # normalize these vectors to produce unit vectors.
    ng_up = galactocentric_up / galactocentric_up.norm()
    ng_north = galactocentric_north / galactocentric_north.norm()
    ng_east = galactocentric_east / galactocentric_east.norm()

    return ng_north, ng_east, ng_up


def plot_dm_component(time=None):
    """
    A function to plot the x and y components of average halo velocity.
    Also gives a nice example of computing a bunch of time locations in one
    step. returns the DM velcoity, as well as the x, y, and z unit vectors,
    all in the galactocentric frame, for convenience.
    """
    if time is None:
        time = Time.now()

    times = np.arange(1535018948.6150093, 1535018948.6150093+60*60*24*365, 600)
    t = Time(times, format='unix')

    halo_vel = get_halo_vel(t)
    x, y, z = get_CASPEr_vect(t)

    vx = x.dot(halo_vel)
    vy = y.dot(halo_vel)

    #plt.plot(vx)
    #plt.plot(vy)
    return halo_vel, x, y, z

