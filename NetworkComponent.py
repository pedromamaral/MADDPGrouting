from environmental_variables import NR_ACTIVE_CONNECTIONS
from Link import Link

class NetworkComponent:


    def __init__(self, id, communication_sequence):

        self.id = id
        self.links = {}
        self.neighbors = []
        self.type = "Host" if id[0] == 'H' else 'Switch'
        self.turns_busy = 0
        self.active_paths = {}
        self.dsts = communication_sequence
        self.current_index = 0
        self.done = False
        self.active_communication_bw = 0
        self.current_dst = None
        self.active_communications = [[-1, -1] for i in range(NR_ACTIVE_CONNECTIONS)]
        self.active_dst = -1
        self.bw_pct = 0


    # adds a link and will add the other component to the neighbors list
    def add_link(self, link:Link):
        self.links[link.get_id()] = link
        components = link.get_components()
        self.neighbors.append([c for c in components if c != self.id][0])


    def add_active_communication(self, origin,destiny):
        for communicaton in self.active_communications:
            if communicaton[0] == -1:
                communicaton[0] = origin
                communicaton[1] = destiny
                return

    def remove_active_communication(self, origin, destiny):
        for communication in self.active_communications:
            if communication[0] == origin and communication[1] == destiny:
                communication[0] = -1
                communication[1] = -1
                return


    def get_active_communications(self):
        communications = []
        for communication in self.active_communications:
            if communication[0] != -1:
                communications.append([int(communication[0][1:]), int(communication[1][1:])])
            else:
                communications.append([-1, -1])
        return communications
        return self.active_communications


    def get_neighbors_bw(self):
        bws = []

        for neighbor in self.neighbors:

            link = self.links.get((self.id, neighbor), None)

            if not link:
                link = self.links.get((neighbor, self.id), None)

            bws.append(link.get_bw_available_percentage())
        return min(bws)

    def add_neighbor(self, host):
        self.neighbors.append(host)
        # TENHO QUE MUDAR DE 0 PARA UM DEFAULT

    def get_link(self, dst):

        if (self.id, dst) in self.links:
            return self.links.get((self.id, dst))
        elif (dst, self.id) in self.links:
            return self.links.get((dst, self.id))
        return None

    def set_communication(self, nr_turns, bw, dst):
        self.turns_busy = nr_turns
        self.active_communication_bw = bw
        self.current_dst = dst

    def update_communication(self):
        self.turns_busy -= 1

    def is_busy(self):
        return self.turns_busy != 0

    def set_active_path(self, dst, id):
        self.active_paths[dst] = id

    def get_active_path(self, dst):
        return self.active_paths.get(dst, None)

    def get_dst(self):
        if self.current_index < len(self.dsts):
            dst = self.dsts[self.current_index]
            self.current_index += 1
            return dst
        self.done = True
        return None

    def get_next_dst(self):
        if self.current_index < len(self.dsts):
            dst = self.dsts[self.current_index]
            return dst
        return None

    def is_done(self):
        return self.done and self.turns_busy == 0

        """
        links = [link for link in self.links if link.check_link(self.id, dst)]
        if links:
            return links[0]
        return None
        """