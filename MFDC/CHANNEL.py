import math

a = 1.38e-23                    # Boltzmann’s constant

class Channel():

    def __init__(self, snode, dnode, type=1):
        self.type = 1                   # 1 for intraplane ISL and 0 for interplane ISL
        self.bandwidth = 1.0            # in GHz
        self.rate = 0.0
        self.frequency = 23e9         # in Hz
        self.noise = 10 * math.log10(a * 300 * self.bandwidth * pow(10, 9))      # in dB
        self.propdelay = 0.0
        self.transdelay = 0.0
        self.snode = snode
        self.dnode = dnode
        self.length = (math.sqrt(math.pow((self.snode.position[0] * 1000 - self.dnode.position[0] * 1000), 2) \
                                + math.pow((self.snode.position[1] * 1000 - self.dnode.position[1] * 1000), 2) \
                                + math.pow((self.snode.position[2] * 1000 - self.dnode.position[2] * 1000), 2)))/1000000
        self.spaceloss = 10 * math.log10(pow(4 * math.pi * self.length * 1000000 * self.frequency / 299792458.0, 2)) # in dB
        self.calc_recv_power()
        self.calc_rate()
        self.calc_propdelay()

    def calc_recv_power(self):
        self.dnode.Pr = self.snode.Pt + self.snode.Gt + self.dnode.Gr - self.spaceloss          # in dBW
        #print(self.dnode.Pr)

    def calc_rate(self):
        self.rate = self.bandwidth * pow(10, 9) * math.log2(1 + pow(10, self.dnode.Pr/10) / pow(10, self.noise/10)) / pow(10, 6)     # in Mbps

    def calc_transdelay(self, package):
        self.transdelay = package.size / self.rate                  # in s
        return self.transdelay

    def calc_propdelay(self):
        self.propdelay = self.length * 1000000 / 299792458.0        # in s
        return self.propdelay


# class Path():
#
#     def __int__(self):
#         super(Path, self).__int__()
#
#     def createPath(self, action):
#         path = []
#         for i in range(len(action)-1):
#             src = action[i]
#             dst = action[i+1]
#             path.append(Channel(src, dst))
#         return path
