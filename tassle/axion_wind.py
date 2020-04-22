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
    # transform to the galactocentric frame and take the velocity, also
    # toss a negative on to get the axion wind speed
    return -CASPEr_inst_loc.galactocentric.cartesian.differentials['s']

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

    This is a function to compute the average velocity and direction of
    the earth within the galactic dark matter halo. The reason is to compute
    the velocity and the direction of the axion wind at the CASPEr detector
    (although this will be the average wind, not the instantaneous wind which
    will vary from one coherence time to another).
    It also computes the unit vectors in the north, east, and up directions at
    the specified time and location, but in the galactocentric frame.

    It accepts optional inputs time, lat, lon, and elev

    time should be a astropy Time object (possibly specifying an array of
    times, in which case the Cartesian representation returned will contain
    an array of locations as well). If not specified, the current time is used.

    lat, lon, and elev specify a location on earth. elev should be in meters.
    By default, the location of CASPEr in Mainz is used.

    Returns (time, unit_x, unit_y, unit_z, V)
    with units (unix time, dimensionless, and Km/sec)
    """
    # if we don't have a time array passed in, just use right now
    if time is None:
        time = Time.now()
    # location on earth of CASPEr
    CASPEr_loc = coord.EarthLocation(lon=lon, lat=lat,
                                     height=elev)
    # location in the solar system of CASPEr at a given moment in GCRS coords.
    CASPEr_inst_loc = coord.SkyCoord(CASPEr_loc.get_gcrs(time))
    # get velocities too, so we can do an all-in-one function
    # note the negative, since we want the velocity of the DM halo on earth
    halo_vels = -CASPEr_inst_loc.galactocentric.cartesian.differentials['s']

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

    return time, ng_north, ng_east, ng_up, halo_vels


def plot_dm_component(days=365, samp_hrs=1, time=None):
    """
    A function to plot the x and y components of average halo velocity.
    Also gives a nice example of computing a bunch of time locations in one
    step. returns the DM velcoity, as well as the x, y, and z unit vectors,
    all in the galactocentric frame, for convenience.
    """
    if time is None:
        time = Time.now().unix

    times = np.arange(time, time + 60*60*24*days, 60*60*samp_hrs)
    t = Time(times, format='unix')

    t, x, y, z, halo_vel = get_CASPEr_vect(t)

    vx = x.dot(halo_vel)
    vy = y.dot(halo_vel)

    plt.plot(vx)
    plt.plot(vy)
    return(t, x, y, z, halo_vel)

############################
##### Nate's Functions #####
############################

def FracDay(Y=2000, M=1, D=1):
    """
    A function that returns the fractional amount of days since Jan 1, 2000
    as calculated by https://arxiv.org/pdf/1312.1355.pdf appendix. 
    """


    #[DD] = FracDay(Y,M,D) - Returns Fractional day (since Jan 2000 at 12.00), as calculated in
    #https://arxiv.org/pdf/1312.1355.pdf appendix A. 
    if (M==1 or M==2):
        YY = Y-1
        MM = M+12
    else:
        YY = Y
        MM = M
    DD = np.floor(365.25*YY) + np.floor(30.60001*(MM+1)) + D - 730563.5 + 0.5

    return DD

def ACalcV(T=0)
    """
    A function that returns the velocity vector in the lab frame (North, East, +Z) in km s^-1
    for an input time(of fractional days since Jan 1, 2000)
    as calculated by https://arxiv.org/pdf/1806.05927 appendix. 
    """

    #For when should the velocity be calculated? In fractional days since Jan 1 2000
    
    #Initial time
    D = 1
    M = 1
    Y = 2000
    
    #Lab
    #Mainz Staudingerweg 18
    lat =  49.9916 * np.pi/180
    lon =  08.2353 * np.pi/180

    #%Times
    DD = FracDay(Y,M,D)
    DD = DD + T

    #%0.997269566666667
    td = ((2*np.pi)/0.99727)*(DD - 0.721)+lon #%Local Apparent Sidereal Time
    #%td = 2*pi*DD/0.9973+lon;
    ty = ((2*np.pi/365)*(DD - FracDay(Y=2000,M=3,D=20.5)))

    #%Test (6 pm GMT 31 Jan 2009 should be DD = 3318.25)
    #%D = 31 + 18/24;
    #%M = 01;
    #%Y = 2009;

    #%Calculate fractional day

    #%Sun velocity in galactic coords (V_LSR + Vpeculiar)
    vsun = 232.6; #km s^-1
    Vsun = vsun*np.array([0.0477, 0.9984, 0.0312]);

    #%Earth revolution velocity in galactic coords
    vearth = 29.79;#km s^-1
    wy = 2*np.pi/365.25; #ty = FracDay(2000,3,20.5);
    e1 = np.array([0.9940, 0.1095, 0.0031])
    e2 = np.array([-0.0517, 0.4945, -0.8677])

    Vearth = vearth*(np.cos(ty)*e1 + np.sin(ty)*e2);

    #%Earth rotation in lab frame
    vrot = 0.47; #km s^-1
    Vrot = vrot*np.cos(lat)*np.array([0, -1, 0]) #always points East

    #%Matrix transformations from galactic to intermediate equatorial:
    Rgal = np.matrix([[-0.05487556, +0.49410943, -0.86766615],
            [-0.87343709, -0.44482963, -0.19807637],
            [-0.48383502, +0.74698225, +0.45598378]])
        
    Rlab = np.matrix([[-np.sin(lat)*np.cos(td), -np.sin(lat)*np.sin(td), np.cos(lat)],
                      [np.sin(td)             , -np.cos(td)            , 0          ],
                      [np.cos(lat)*np.cos(td) , np.cos(lat)*np.sin(td) , np.sin(lat)]])

    v = (Rlab.dot(Rgal)).dot(Vsun + Vearth) + (Vrot);
    
return v


###################################
##### End of Nate's Functions #####
###################################
