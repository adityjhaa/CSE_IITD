from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet
from ryu.lib.packet import ethernet
from ryu.lib.packet import ether_types
from ryu.topology.api import get_switch, get_link
from ryu.topology import event

class Spanning_Tree_Switch(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(Spanning_Tree_Switch, self).__init__(*args, **kwargs)
        self.mac_to_port = {}
        self.topology = {}
        self.root_bridge = None
        self.spanning_tree = set()
        self.blocked_ports = set()
        self.datapaths = {}

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        """Set up the controller to handle packets."""
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                          ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)
        self.datapaths[datapath.id] = datapath

    @set_ev_cls(event.EventSwitchEnter)
    def switch_enter_handler(self, ev):
        """Trigger topology update once switches are detected."""
        self.datapaths[ev.switch.dp.id] = ev.switch.dp
        self.update_topology()
    
    def update_topology(self):
        """Updates the network topology and runs the spanning tree algorithm."""
        self.topology.clear()
        switches = get_switch(self, None)
        links = get_link(self, None)

        for switch in switches:
            dpid = switch.dp.id
            self.topology.setdefault(dpid, set())

        for link in links:
            src_dpid = link.src.dpid
            dst_dpid = link.dst.dpid
            src_port = link.src.port_no
            dst_port = link.dst.port_no

            self.topology[src_dpid].add((dst_dpid, src_port))
            self.topology[dst_dpid].add((src_dpid, dst_port))

        # self.logger.info(f"Updated topology: {self.topology}")
        self.run_stp()

    def run_stp(self):
        """Constructs a spanning tree using DFS."""
        if not self.topology:
            return

        self.root_bridge = min(self.topology.keys())
        visited = set()

        self.spanning_tree.clear()
        self.dfs(self.root_bridge, visited)

        # self.logger.info(f"Constructed Spanning Tree: {self.spanning_tree}")

        self.block_non_tree_ports()

    def dfs(self, switch, visited):
        """Performs DFS to build the spanning tree."""
        visited.add(switch)
        for neighbor, port in self.topology[switch]:
            if neighbor not in visited:
                self.spanning_tree.add((switch, neighbor))
                self.spanning_tree.add((neighbor, switch))
                self.dfs(neighbor, visited)

    def block_non_tree_ports(self):
        """Blocks the ports that are not part of the spanning tree."""
        for switch in self.topology:
            for neighbor, port in self.topology[switch]:
                if (switch, neighbor) not in self.spanning_tree:
                    self.blocked_ports.add((switch, port))
                    self.block_port(switch, port)

    def block_port(self, dpid, port):
        """Inserts a flow to block traffic through a specific port."""
        if dpid in self.datapaths:
            datapath = self.datapaths[dpid]
            ofproto = datapath.ofproto
            parser = datapath.ofproto_parser

            match = parser.OFPMatch(in_port=port)
            actions = []
            self.add_flow(datapath, priority=5, match=match, actions=actions)

    def add_flow(self, datapath, priority, match, actions, buffer_id=None):
        """Adds a flow entry to a switch."""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        if buffer_id:
            mod = parser.OFPFlowMod(datapath=datapath, buffer_id=buffer_id,
                                    priority=priority, match=match, instructions=inst)
        else:
            mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                    match=match, instructions=inst)
        datapath.send_msg(mod)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        """Handles packets coming into the switch."""
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]

        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            # Ignore LLDP packets
            return
        dst = eth.dst
        src = eth.src

        dpid = datapath.id
        self.mac_to_port.setdefault(dpid, {})

        self.mac_to_port[dpid][src] = in_port

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

