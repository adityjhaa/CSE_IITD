from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ether_types, lldp
from ryu.topology.api import get_switch, get_link
from ryu.topology import event
from ryu.lib import hub
from collections import defaultdict
import time

class Graph:
    def __init__(self):
        self.nodes = set()
        self.topology = defaultdict(dict)
        self.delays = {}

    def add_node(self, value):
        self.nodes.add(value)

    def add_edge(self, from_node, to_node, from_port, to_port, delay=float('inf')):
        self.topology[from_node][to_node] = from_port
        self.topology[to_node][from_node] = to_port
        self.delays[(from_node, to_node)] = delay
        self.delays[(to_node, from_node)] = delay
        
    def __repr__(self):
        node_repr = "Nodes: " + ", ".join(map(str, self.nodes))
        edge_repr = "Edges with Delays:\n" + "\n".join(
            f"{src} -> {dst}: {delay} ms" for (src, dst), delay in self.delays.items()
        )
        return f"{node_repr}\n{edge_repr}"

class DelayBasedShortestPathRouting(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(DelayBasedShortestPathRouting, self).__init__(*args, **kwargs)
        self.network = Graph()
        self.switches = {}
        self.mac_to_port = {}
        self.dpid_to_mac = {}
        self.switch_ports = {}
        self.lldp_delays = {}
        self.datapaths = {}
        self.is_topology_discovered = False
        self.topology_discovery_complete = hub.Event()
        self.lldp_probe_complete = hub.Event()
        
    def start(self):
        super(DelayBasedShortestPathRouting, self).start()
        self.threads.append(hub.spawn(self._topology_discover))

    def _topology_discover(self):
        while True:
            self.get_topology_data()
            hub.sleep(10)  # Wait for 10 seconds before rediscovering topology
            self.topology_discovery_complete.clear()
            self.lldp_probe_complete.clear()

    def get_topology_data(self):
        switch_list = get_switch(self)
        links_list = get_link(self)
        
        self.network = Graph()  # Reset the network graph
        self.switches.clear()
        self.switch_ports.clear()
        
        for sw in switch_list:
            self.network.add_node(sw.dp.id)
            self.switches[sw.dp.id] = sw.dp
        
        for link in links_list:
            src = link.src
            dst = link.dst
            self.network.add_edge(src.dpid, dst.dpid, src.port_no, dst.port_no)
            self.switch_ports.setdefault(src.dpid, {})[dst.dpid] = src.port_no
            self.switch_ports.setdefault(dst.dpid, {})[src.dpid] = dst.port_no

        self.topology_discovery_complete.set()
        hub.spawn(self._probe_links)

    def _probe_links(self):
        self.topology_discovery_complete.wait()
        self.send_lldp_packets()
        hub.sleep(5)  # Wait for LLDP packets to be received
        self.lldp_probe_complete.set()
        hub.spawn(self._update_paths)

    def _update_paths(self):
        self.lldp_probe_complete.wait()
        self.update_topology()

    def send_lldp_packets(self):
        for dp in self.switches.values():
            if dp.id not in self.switch_ports:
                continue
            for port in self.switch_ports[dp.id].values():
                self.send_lldp_packet(dp, port)

    def send_lldp_packet(self, datapath, port):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        pkt = packet.Packet()
        pkt.add_protocol(ethernet.ethernet(
            ethertype=ether_types.ETH_TYPE_LLDP,
            src=datapath.ports[port].hw_addr,
            dst=lldp.LLDP_MAC_NEAREST_BRIDGE
        ))
        pkt.add_protocol(lldp.lldp(tlvs=[
            lldp.ChassisID(subtype=lldp.ChassisID.SUB_LOCALLY_ASSIGNED, chassis_id=str(datapath.id).encode('ascii')),
            lldp.PortID(subtype=lldp.PortID.SUB_PORT_COMPONENT, port_id=str(port).encode('ascii')),
            lldp.TTL(ttl=120),
        ]))
        pkt.serialize()
        data = pkt.data
        actions = [parser.OFPActionOutput(port)]
        out = parser.OFPPacketOut(
            datapath=datapath,
            buffer_id=ofproto.OFP_NO_BUFFER,
            in_port=ofproto.OFPP_CONTROLLER,
            actions=actions,
            data=data
        )

        datapath.send_msg(out)
        self.lldp_delays[(datapath.id, port)] = time.time()

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                          ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)
        self.datapaths[datapath.id] = datapath

    def add_flow(self, datapath, priority, match, actions, buffer_id=None):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        if buffer_id:
            mod = parser.OFPFlowMod(
                datapath=datapath,
                buffer_id=buffer_id,
                priority=priority,
                match=match,
                instructions=inst
            )
        else:
            mod = parser.OFPFlowMod(
                datapath=datapath,
                priority=priority,
                match=match,
                instructions=inst
            )
        datapath.send_msg(mod)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]

        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            self.handle_lldp(datapath, in_port, pkt)
            return

        dst = eth.dst
        src = eth.src

        dpid = datapath.id
        self.mac_to_port.setdefault(dpid, {})
        self.mac_to_port[dpid][src] = in_port
        self.dpid_to_mac[dpid] = src

        if dst in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst]
        else:
            out_port = ofproto.OFPP_FLOOD

        actions = [parser.OFPActionOutput(out_port)]

        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst, eth_src=src)
            if msg.buffer_id != ofproto.OFP_NO_BUFFER:
                self.add_flow(datapath, 1, match, actions, msg.buffer_id)
                return
            else:
                self.add_flow(datapath, 1, match, actions)

        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data

        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                  in_port=in_port, actions=actions, data=data)
        datapath.send_msg(out)

    def handle_lldp(self, datapath, port, pkt):
        timestamp = time.time()
        lldp_pkt = pkt.get_protocol(lldp.lldp)
        
        if lldp_pkt:
            src_dpid_str = (lldp_pkt.tlvs[0].chassis_id.decode('ascii'))
            if src_dpid_str.isdigit():
                src_dpid = int(src_dpid_str)
            else:
                src_dpid = int(src_dpid_str.split(':')[-1], 16)
            src_port = int.from_bytes(lldp_pkt.tlvs[1].port_id, byteorder='big')
            dst_dpid = datapath.id
            dst_port = port

            if (src_dpid, src_port) in self.lldp_delays:
                delay = (timestamp - self.lldp_delays[(src_dpid, src_port)]) / 2
                self.network.add_edge(src_dpid, dst_dpid, src_port, dst_port, delay)

    @set_ev_cls(event.EventSwitchEnter)
    def switch_enter_handler(self, ev):
        if not self.is_topology_discovered:
            self.is_topology_discovered = True
            hub.spawn(self._topology_discover)

    def update_topology(self):
        self.install_paths()

    def install_paths(self):
        for src in self.network.nodes:
            for dst in self.network.nodes:
                if src != dst:
                    path = self.shortest_path(src, dst)
                    if path:
                        self.install_path_flows(path)
                    else:
                        print(f"No path found from {src} to {dst}")

    def shortest_path(self, src, dst):
        distances = {src: 0}
        previous = {}
        nodes = list(self.network.nodes)

        while nodes:
            current = min(nodes, key=lambda node: distances.get(node, float('inf')))
            nodes.remove(current)

            if current == dst:
                path = []
                while current in previous:
                    path.append(current)
                    current = previous[current]
                path.append(src)
                return list(reversed(path))

            for neighbor, _ in self.network.topology[current].items():
                alt = distances[current] + self.network.delays.get((current, neighbor), float('inf'))
                if alt < distances.get(neighbor, float('inf')):
                    distances[neighbor] = alt
                    previous[neighbor] = current

        return None

    def install_path_flows(self, path):
        for i in range(len(path) - 1):
            src_dpid = path[i]
            dst_dpid = path[-1]  # Always use the final destination
            next_hop = path[i + 1]
            
            if src_dpid not in self.switch_ports or next_hop not in self.switch_ports[src_dpid]:
                print(f"Missing port information for {src_dpid} -> {next_hop}")
                continue
            
            out_port = self.switch_ports[src_dpid][next_hop]
            
            datapath = self.switches[src_dpid]
            parser = datapath.ofproto_parser
            actions = [parser.OFPActionOutput(out_port)]
            
            dst_mac = self.dpid_to_mac.get(dst_dpid)
            if not dst_mac:
                print(f"Missing MAC address for switch {dst_dpid}")
                continue
            
            match = parser.OFPMatch(eth_dst=dst_mac)
            
            self.add_flow(datapath, 2, match, actions)
