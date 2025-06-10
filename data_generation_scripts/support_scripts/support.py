from pydylan import Body
from pydylan.eom import S2BP, CR3BP

from numpy import array

html_colors = {'baby blue': '#33D7FF',
               'strong orange': '#FF6400',
               'light green': '#40FF00'}

def get_GTO_in_PR3BP_units():
    earth = Body("Earth")
    moon = Body("Moon")
    s2bp = S2BP(earth)
    cr3bp = CR3BP(earth, moon)
    state = s2bp.coe2rv(24510.,
                        0.72345981,
                        0.,
                        0.,
                        0.,
                        3.141592)

    position = array(state[0])
    velocity = array(state[1])

    return array([-cr3bp.mu + position[0] / cr3bp.DU,
                  position[1] / cr3bp.DU,
                  0.,
                  velocity[0] / cr3bp.VU,
                  velocity[1] / cr3bp.VU,
                  0.])


def get_LLO_in_PR3BP_units():
    earth = Body("Earth")
    moon = Body("Moon")
    s2bp = S2BP(moon)
    cr3bp = CR3BP(earth, moon)

    # LLO: 1000km circular orbit
    state = s2bp.coe2rv(1000. + moon.radius,
                        0.,
                        0.,
                        0.,
                        0.,
                        0.)

    position = array(state[0])
    velocity = array(state[1])

    return array([1.0 - cr3bp.mu + position[0] / cr3bp.DU,
                  position[1] / cr3bp.DU,
                  0.0,
                  -velocity[0] / cr3bp.VU,
                  -velocity[1] / cr3bp.VU,
                  0.0])


def get_GTO_in_CR3BP_units():
    earth = Body("Earth")
    moon = Body("Moon")
    s2bp = S2BP(earth)
    cr3bp = CR3BP(earth, moon)
    state = s2bp.coe2rv(24510.,
                        0.72345981,
                        (15.0 * 3.141592) / 180.0,
                        0.,
                        0.,
                        3.141592)

    position = array(state[0])
    velocity = array(state[1])

    return array([-cr3bp.mu + position[0] / cr3bp.DU,
                  position[1] / cr3bp.DU,
                  position[2] / cr3bp.DU,
                  velocity[0] / cr3bp.VU,
                  velocity[1] / cr3bp.VU,
                  velocity[2] / cr3bp.VU])


def get_LLO_in_CR3BP_units(altitude: float = 10000.):
    earth = Body("Earth")
    moon = Body("Moon")
    s2bp = S2BP(moon)
    cr3bp = CR3BP(earth, moon)

    # LLO: 1000km circular orbit
    state = s2bp.coe2rv(altitude + moon.radius,
                        0.,
                        (195. * 3.141592) / 180.,
                        3.141592,
                        0.,
                        0.)

    position = array(state[0])
    velocity = array(state[1])

    return array([1.0 - cr3bp.mu + position[0] / cr3bp.DU,
                  position[1] / cr3bp.DU,
                  position[2] / cr3bp.DU,
                  -velocity[0] / cr3bp.VU,
                  -velocity[1] / cr3bp.VU,
                  -velocity[2] / cr3bp.VU])
