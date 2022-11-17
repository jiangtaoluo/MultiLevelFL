import math
from skyfield.api import Loader, EarthSatellite, wgs84
from skyfield_data import get_skyfield_data_path

load = Loader(get_skyfield_data_path())
planets = load('de421.bsp')
ts = load.timescale(builtin=False)
frequency = 23e9            # ware frequency, in Hz
c = 299792458.0             # light speed, in m/s
l = c / frequency           # the ware length, in m

class Node():

    def __init__(self, name, index, line1, line2):
        self.position = [0, 0, 0]
        self.name = name
        self.power = 100.0                # in Wh, the initial power
        self.area = 0.5                # in m^2, the area of solar panel
        self.gamma = 1353               # in W/m^2, the irradiance per unit area
        self.eta1 = 0.15                 # solar ---> electricity efficiency
        self.alpha = 0                  # the angle between sunlight and the normal direction of solar panel
        self.harvest_per_time = self.area * self.gamma * self.eta1 * math.cos(self.alpha) / (60 * 60)     # in Wh
        self.D = 0.5                    # in m, the diameter of antenna
        self.Gt = 10 * math.log10(4 * math.pi * math.pow(self.D, 2) / math.pow(l, 2))       # in dB
        self.eta2 = 0.65                # the antenna efficiency
        self.Gr = 10 * math.log10(self.eta2 * math.pow(math.pi * self.D / l, 2))            # in dB
        self.Pt = 10 * math.log10(1)                     # in dBW
        self.Pr = 0.0
        self.PrMIN = -150               # in dBW
        self.index = index
        self.line1 = line1
        self.line2 = line2
        self.P0 = 100                     # in W, the constant power consumption
        self.rho1 = 0.05                # in W/Mbps, for sending
        self.rho2 = 0.01                # in W/Mbps, for receiving
        self.rho3 = 0.01                # in W/Mbps, for table lookup
        self.mu = 0.01                  # in W/Mbps, for processor
        self.alpha_ = 1.4               # constant for energy consumption
        self.drop = 0
        self.channel = []
        self.neighbor = []
        self.lat = 0
        self.sat = EarthSatellite(self.line1, self.line2, self.name, ts)

    def setposition_lat(self, t):
        '''assert(len(pos) == 3)
        for i in range(3):
            self.position[i] = pos[i]'''
        t1 = ts.utc(t[0], t[1], t[2], t[3], t[4], t[5])
        self.geocentric = self.sat.at(t1)
        self.position = self.geocentric.position.km
        subpoint = wgs84.subpoint(self.geocentric)
        self.lat = subpoint.latitude

    def harvester(self, t1):
        sunlit = self.sat.at(ts.utc(t1[0], t1[1], t1[2], t1[3], t1[4], t1[5])).is_sunlit(planets)
        if sunlit and self.power < 100 - self.harvest_per_time:
            energy_harvested = self.harvest_per_time
            self.power += energy_harvested
        #return energy_harvested

    def consume_energy(self, p, path):
        if int(self.name) in path:
            if int(self.name) == path[0]:
                power_consumed = (self.P0 + p.size * (self.rho3 + self.rho1) +
                                  self.mu * math.pow(p.size, self.alpha_)) / (60 * 60)
                self.power -= power_consumed
            elif int(self.name) == path[-1]:
                power_consumed = (self.P0 + p.size * self.rho2 +
                                  self.mu * math.pow(p.size, self.alpha_)) / (60 * 60)
                self.power -= power_consumed
            else:
                power_consumed = (self.P0 + p.size * (self.rho3 + self.rho1 + self.rho2) +
                                  self.mu * math.pow(p.size, self.alpha_)) / (60 * 60)
                self.power -= power_consumed
        else:
            power_consumed = self.P0 / (60 * 60)
            self.power -= power_consumed
        return power_consumed

    def setname(self, name):
        self.name = name

    def setindex(self, index):
        self.index = index

    def setline1(self, tle1):
        self.line1 = tle1

    def setline2(self, tle2):
        self.line2 = tle2

    def addchannel(self, channel):
        self.channel.append(channel)

    def isdropped(self, package):
        if self.power < package.energy_required:
            self.drop = 1