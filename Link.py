class Link:

    def __init__(self, src, dst, bw):
        self.src = src
        self.dst = dst
        self.bw_total = int(bw)
        self.bw_available = self.bw_total
        self.bw_used = 0
        self.active_connections = {}

    def get_id(self):
        return (self.src, self.dst)

    def get_components(self):
        return [self.src, self.dst]

    # returns True or False if its link between src and dst
    def check_link(self, src, dst):
        return (self.src == src and self.dst == dst) or (self.src == dst and self.dst == src)

    def update_bw(self, used_bw):
        self.bw_available -= used_bw
        self.bw_used += used_bw

    def get_bw_available_percentage(self):
        return self.bw_available / self.bw_total * 100

    def add_active_communication(self, origin, destiny, bw):
        self.active_connections[(origin, destiny)] = bw

    def get_active_communication(self, origin, destiny):
        return self.active_connections.get((origin, destiny), 0)