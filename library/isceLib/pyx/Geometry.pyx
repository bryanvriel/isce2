#cython: language_level=3

import numpy as np
cimport numpy as np

def geo2rdr(PyOrbit orbit, PyEllipsoid ellps, list llh):
    """
    Performs geo2rdr assuming zero Doppler.
    """

    # Convert llh -> xyz
    cdef list xyz = [0.0, 0.0, 0.0]
    ellps.latLon(xyz, llh, 1)

    # Initial guess for azimuth time
    cdef int nVectors = orbit.nVectors
    cdef double tspan = orbit.c_orbit.UTCtime[nVectors-1] - orbit.c_orbit.UTCtime[0]
    cdef double tguess = orbit.c_orbit.UTCtime[0] + 0.5 * tspan

    # Constant doppler terms
    cdef double fdop = 0.0
    cdef double fdopder = 0.0

    # Initialize temporary variables
    cdef list satpos = [0.0, 0.0, 0.0]
    cdef list satvel = [0.0, 0.0, 0.0]
    cdef list dr = [0.0, 0.0, 0.0]
    cdef double rng
    cdef double dopfact
    cdef double fn
    cdef double c1
    cdef double c2
    cdef double fnprime
    cdef double delta_seconds

    # Perform iterations
    cdef int i
    cdef int ii
    cdef int orbstat
    for ii in range(51):

        # Interpolate orbit to azimuth time
        orbstat = orbit.interpolateWGS84Orbit(tguess, satpos, satvel)

        # Compute slant range
        for i in range(3):
            dr[i] = xyz[i] - satpos[i]
        rng = np.sqrt(dr[0]**2 + dr[1]**2 + dr[2]**2)

        # Doppler factors
        dopfact = 0.0
        for i in range(3):
            dopfact += dr[i] * satvel[i]

        # Update steps
        fn = dopfact - fdop * rng
        c1 = -1.0 * (satvel[0]**2 + satvel[1]**2 + satvel[2]**2)
        c2 = fdop / rng + fdopder
        fnprime = c1 + c2 * dopfact

        # Check convergence
        delta_seconds = fn / fnprime
        if abs(delta_seconds) < 1.0e-6:
            break

        # Update time
        tguess -= delta_seconds
        
    return tguess, rng


def rdr2geo(double aztime, double rng, PyOrbit orbit, PyEllipsoid ellps,
            double height=0.0, double side=-1.0):

    cdef list satpos = [0.0, 0.0, 0.0]
    cdef list satvel = [0.0, 0.0, 0.0]
    orbit.interpolateWGS84Orbit(aztime, satpos, satvel)

    # Convert position and velocity to local tangent plane
    cdef double major = ellps.a
    cdef double minor = major * np.sqrt(1.0 - ellps.e2)
    cdef list posnorm = [satpos[0]/major, satpos[1]/major, satpos[2]/minor]

    # Set up ortho normal system right below satellite
    cdef double satDist = norm(satpos)
    cdef double alpha = 1.0 / norm(posnorm)
    cdef double radius = alpha * satDist
    cdef double hgt = (1.0 - alpha) * satDist

    # Set up TCN basic (geocentric)
    cdef temp = [0.0, 0.0, 0.0]
    cdef list chat = [0.0, 0.0, 0.0]
    cdef list that = [0.0, 0.0, 0.0]
    cdef list vhat = [0.0, 0.0, 0.0]
    cdef list nhat = [-satpos[0]/satDist, -satpos[1]/satDist, -satpos[2]/satDist]
    cross(nhat, satvel, temp)
    unitVec(temp, chat)
    cross(chat, nhat, temp)
    unitVec(temp, that)    
    unitVec(satvel, vhat)
   
    # Initial guess
    cdef double zsch = height

    # Perform iterations
    cdef double a
    cdef double b
    cdef double costheta
    cdef double sintheta
    cdef double gamma
    cdef double beta
    cdef double delta
    
    cdef list targVec = [0.0, 0.0, 0.0]
    cdef list targLLH = [0.0, 0.0, 0.0]
    cdef list tempLLH = [0.0, 0.0, 0.0]
    cdef list targXYZ = [0.0, 0.0, 0.0]

    cdef double gamma_fact = dot(nhat, vhat) / dot(vhat, that)
    cdef double dopfact = 0.0

    cdef int ii
    cdef double rdiff
    for ii in range(10):

        a = satDist
        b = radius + zsch

        costheta = 0.5 * (a/rng + rng/a - (b/a)*(b/rng))
        sintheta = np.sqrt(1-costheta*costheta)

        gamma = rng * costheta
        alpha = dopfact - gamma * gamma_fact
        beta = -side * np.sqrt(rng*rng*sintheta*sintheta - alpha*alpha)

        for i in range(3):
            delta = alpha * that[i] + beta * chat[i] + gamma * nhat[i]
            targVec[i] = satpos[i] + delta

        ellps.latLon(targVec, targLLH, 2)

        tempLLH[0] = targLLH[0]
        tempLLH[1] = targLLH[1]
        tempLLH[2] = height
        
        ellps.latLon(targXYZ, tempLLH, 1)
        
        zsch = norm(targXYZ) - radius
       
        for i in range(3):
            temp[i] = satpos[i] - targXYZ[i]

        rdiff = rng - norm(temp)
        if abs(rdiff) < 1.0e-5:
            break

    return targLLH

cdef norm(list v):
    return np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)

cdef unitVec(list v, list v_unit):
    cdef double vnorm = norm(v)
    cdef int i
    for i in range(3):
        v_unit[i] = v[i] / vnorm

cdef dot(list a, list b):
    cdef int i
    cdef double result = 0.0
    for i in range(3):
        result += a[i] * b[i]
    return result

cdef cross(list a, list b, list c):

    cdef double _a[3]
    cdef double _b[3]
    cdef double _c[3]
    cdef int i
    for i in range(3):
        _a[i] = a[i]
        _b[i] = b[i]

    cdef LinAlg c_linAlg
    c_linAlg.cross(_a, _b, _c)

    for i in range(3):
        c[i] = _c[i]

# end of file
