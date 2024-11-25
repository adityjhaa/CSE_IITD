from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ether_types, lldp
from ryu.topology.api import get_switch, get_link
from ryu.topology import event
from ryu.lib import hub
from collections import defaultdict
import heapq
import time
import threading

class Graph:
    def __init__(self):
        self.nodes = set()
        self.edges = defaultdict(dict)
        self.delays = {}

    def add_node(self, value):
        self.nodes.add(value)

    def add_edge(self, from_node, to_node, delay):
        self.edges[from_node][to_node] = delay
        self.edges[to_node][from_node] = delay
        self.delays[(from_node, to_node)] = delay
        self.delays[(to_node, from_node)] = delay
        
    def __repr__(self):
        return str(self.edges)

class CongestionAwareRouting(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(CongestionAwareRouting, self).__init__(*args, **kwargs)
        self.topology = Graph()
        self.switches = {}
        self.mac_to_port = {}
        self.dpid_to_mac = {}
        self.switch_ports = {}
        self.datapaths = {}
        self.lldp_delays = {}
        self.switch_port_updated = {}
        self.topology_ready = False
        self.lldp_ready = False
        self.update_interval = 5
        self.monitor_thread = hub.spawn(self._monitor)
        self.update_topology_thread = hub.spawn(self._update_topology_periodically)

    @set_ev_cls(event.EventSwitchEnter)
    def switch_enter_handler(self, ev):
        self.update_initial_topology()

    def update_initial_topology(self):
        time.sleep(1)
        switch_list = get_switch(self)
        links_list = get_link(self)
        
        for sw in switch_list:
            self.topology.add_node(sw.dp.id)
            self.switches[sw.dp.id] = sw.dp
        
        for link in links_list:
            src = link.src
            dst = link.dst
            
            self.switch_ports.setdefault(src.dpid, {})[dst.dpid] = src.port_no
            self.switch_ports.setdefault(dst.dpid, {})[src.dpid] = dst.port_no
            
            self.switch_port_updated[(src.dpid, src.port_no)] = False
            self.switch_port_updated[(dst.dpid, dst.port_no)] = False
            
            self.topology.add_edge(src.dpid, dst.dpid, float('inf'))
        
        self.topology_ready = True

    def _monitor(self):
        while True:
            for dp in self.datapaths.values():
                self._send_lldp_packet(dp)
            hub.sleep(self.update_interval)

    def _update_topology_periodically(self):
        while True:
            if self.topology_ready and self.lldp_ready:
                self.update_topology()
                self.install_paths()
            hub.sleep(self.update_interval)

    def _send_lldp_packet(self, datapath):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        for port in self.switch_ports.get(datapath.id, {}).values():
            pkt = packet.Packet()
            pkt.add_protocol(ethernet.ethernet(
                ethertype=ether_types.ETH_TYPE_LLDP,
                src=datapath.ports[port].hw_addr,
                dst=lldp.LLDP_MAC_NEAREST_BRIDGE))
            pkt.add_protocol(lldp.lldp(
                tlvs=[lldp.ChassisID(subtype=lldp.ChassisID.SUB_LOCALLY_ASSIGNED,
                                    chassis_id=str(datapath.id).encode('ascii')),
                    lldp.PortID(subtype=lldp.PortID.SUB_PORT_COMPONENT,
                                port_id=str(port).encode('ascii')),
                    lldp.TTL(ttl=120)]))
            pkt.serialize()
            data = pkt.data
            actions = [parser.OFPActionOutput(port)]
            out = parser.OFPPacketOut(datapath=datapath, buffer_id=ofproto.OFP_NO_BUFFER,
                                    in_port=ofproto.OFPP_CONTROLLER,
                                    actions=actions, data=data)
            
            datapath.send_msg(out)
            self.lldp_delays[(datapath.id, port)] = time.time()

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER, ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)
        self.datapaths[datapath.id] = datapath

    def add_flow(self, datapath, priority, match, actions, buffer_id=None):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        if buffer_id:
            mod = parser.OFPFlowMod(datapath=datapath, buffer_id=buffer_id,
                                    priority=priority, match=match,
                                    instructions=inst)
        else:
            mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                    match=match, instructions=inst)
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
            self.handle_lldp(datapath, in_port, pkt, time.time())
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

    def handle_lldp(self, datapath, port, pkt, timestamp):
        lldp_pkt = pkt.get_protocol(lldp.lldp)
        if lldp_pkt:
            try:
                src_dpid_str = (lldp_pkt.tlvs[0].chassis_id.decode('ascii'))
                src_dpid = int(src_dpid_str.split(":")[-1], 16)
                src_port = int.from_bytes(lldp_pkt.tlvs[1].port_id, byteorder='big')
                dst_dpid = datapath.id
                
                if (src_dpid, src_port) in self.lldp_delays:
                    delay = (timestamp - self.lldp_delays[src_dpid, src_port]) * 1000 / 2  # in milliseconds
                    self.topology.add_edge(src_dpid, dst_dpid, delay)
                    self.switch_port_updated[(src_dpid, src_port)] = True
                                        
                    if all(self.switch_port_updated.values()):
                        self.lldp_ready = True
            
            except ValueError:
                return

    def update_topology(self):
        # This method updates the topology based on the latest LLDP measurements
        for (src_dpid, src_port), updated in self.switch_port_updated.items():
            if updated:
                dst_dpid = next((dst for dst, port in self.switch_ports[src_dpid].items() if port == src_port), None)
                if dst_dpid:
                    delay = self.topology.delays.get((src_dpid, dst_dpid), float('inf'))
                    self.topology.add_edge(src_dpid, dst_dpid, delay)
                self.switch_port_updated[(src_dpid, src_port)] = False

    def install_paths(self):
        for src in self.topology.nodes:
            for dst in self.topology.nodes:
                if src != dst:
                    path = self.dijkstra(src, dst)
                    if path:
                        self.install_path_flows(path)

    def dijkstra(self, src, dst):
        distances = {node: float('inf') for node in self.topology.nodes}
        distances[src] = 0
        pq = [(0, src)]
        previous = {}
        while pq:
            current_distance, current_node = heapq.heappop(pq)

            if current_node == dst:
                path = []
                while current_node in previous:
                    path.append(current_node)
                    current_node = previous[current_node]
                path.append(src)
                return list(reversed(path))

            if current_distance > distances[current_node]:
                continue

            for neighbor, delay in self.topology.edges[current_node].items():
                distance = current_distance + delay
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous[neighbor] = current_node
                    heapq.heappush(pq, (distance, neighbor))

        return None

    def install_path_flows(self, path):
        for i in range(len(path) - 1):
            src_dpid = path[i]
            next_hop = path[i + 1]
            
            out_port = self.switch_ports[src_dpid][next_hop]
            datapath = self.switches[src_dpid]
            parser = datapath.ofproto_parser
            
            dst_dpid = path[-1]
            dst_mac = self.dpid_to_mac.get(dst_dpid)
            if not dst_mac:
                continue
            
            actions = [parser.OFPActionOutput(out_port)]
            match = parser.OFPMatch(eth_dst=dst_mac)

            self.add_flow(datapath, 2, match, actions)
