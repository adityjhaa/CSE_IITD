import socket
import time
import argparse
import json

MSS = 1400
DUP_ACK_THRESHOLD = 3
FILE_PATH = "file.txt"

estimated_rtt = 1.0
dev_rtt = 0.5
timeout = estimated_rtt + 4 * dev_rtt

SLOW_START = 'slow_start'
CONGESTION_AVOIDANCE = 'congestion_avoidance'
FAST_RECOVERY = 'fast_recovery'

def update_timeout(sample_rtt, alpha=0.125, beta=0.25):
    """Update timeout value using RTT estimation."""
    global estimated_rtt, dev_rtt, timeout
    estimated_rtt = (1 - alpha) * estimated_rtt + alpha * sample_rtt
    dev_rtt = (1 - beta) * dev_rtt + beta * abs(sample_rtt - estimated_rtt)
    timeout = estimated_rtt + 4 * dev_rtt

def create_packet(seq_num, data):
    """Create a packet with sequence number and data."""
    packet_data = {
        'seq': seq_num,
        'size': len(data),
        'data': data.decode('latin1'),
        'fin': 0
    }
    return json.dumps(packet_data).encode()

def create_end_packet():
    """Create an end-of-transmission packet."""
    packet_data = {
        'seq': -1,
        'size': 0,
        'data': '',
        'fin': 1
    }
    return json.dumps(packet_data).encode()

def handle_new_ack(cwnd, ssthresh, state):
    """Handle receipt of a new ACK."""
    if state == SLOW_START:
        cwnd += MSS
        if cwnd >= ssthresh:
            state = CONGESTION_AVOIDANCE
    
    elif state == CONGESTION_AVOIDANCE:
        cwnd += MSS * (MSS / cwnd)
    
    elif state == FAST_RECOVERY:
        cwnd = ssthresh
        state = CONGESTION_AVOIDANCE
    
    return cwnd, state

def handle_duplicate_ack(cwnd, ssthresh, state, dup_ack_count):
    """Handle receipt of a duplicate ACK."""
    dup_ack_count += 1
    
    if dup_ack_count == DUP_ACK_THRESHOLD:
        ssthresh = max(cwnd // 2, 2 * MSS)
        cwnd = ssthresh + 3 * MSS
        state = FAST_RECOVERY
    
    elif state == FAST_RECOVERY:
        cwnd += MSS
    
    return cwnd, ssthresh, state, dup_ack_count

def handle_timeout(cwnd, ssthresh):
    """Handle timeout event."""
    ssthresh = max(cwnd // 2, 2 * MSS)
    cwnd = MSS
    state = SLOW_START
    return cwnd, ssthresh, state

def send_file(server_ip, server_port):
    """Send a file to the client, ensuring reliability over UDP with congestion control."""
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_socket.bind((server_ip, server_port))
    print(f"Server listening on {server_ip}:{server_port}")

    cwnd = MSS
    ssthresh = 64 * 1024
    state = SLOW_START
    
    client_address = None
    while not client_address:
        try:
            data, client_address = server_socket.recvfrom(1024)
            if data == b"START":
                print(f"Connection established with client {client_address}")
                server_socket.sendto(b"ACK_START", client_address)
        except socket.timeout:
            continue
    
    with open(FILE_PATH, 'rb') as file:
        seq_num = 0
        window_base = 0
        unacked_packets = {}
        duplicate_ack_count = 0
        last_ack_received = -1
        file_finished = False
        end_packet_sent = False

        while True:
            while seq_num < window_base + int(cwnd) // MSS and not file_finished:
                chunk = file.read(MSS)
                if not chunk:
                    file_finished = True
                    break

                packet = create_packet(seq_num, chunk)
                server_socket.sendto(packet, client_address)
                unacked_packets[seq_num] = (packet, time.time())
                seq_num += 1

            if file_finished and not unacked_packets and not end_packet_sent:
                end_packet = create_end_packet()
                server_socket.sendto(end_packet, client_address)
                end_packet_sent = True
                print("Sent END signal")

            try:
                server_socket.settimeout(timeout)
                ack_packet, _ = server_socket.recvfrom(1024)
                
                if ack_packet.decode() == "START":
                    print("Received START signal from client")
                    continue
                
                if not ack_packet:
                    continue
                
                ack_data = json.loads(ack_packet.decode())
                ack_seq_num = ack_data['ack']

                if ack_seq_num == -1 and file_finished:
                    print("Received ACK for END signal")
                    break

                if ack_seq_num > last_ack_received:
                    if ack_seq_num in unacked_packets:
                        sample_rtt = time.time() - unacked_packets[ack_seq_num][1]
                        update_timeout(sample_rtt)
                    
                    last_ack_received = ack_seq_num
                    window_base = ack_seq_num
                    
                    unacked_packets = {seq: pkt for seq, pkt in unacked_packets.items() 
                                    if seq >= window_base}
                    
                    cwnd, state = handle_new_ack(cwnd, ssthresh, state)
                    duplicate_ack_count = 0
                    
                else:
                    cwnd, ssthresh, state, duplicate_ack_count = handle_duplicate_ack(
                        cwnd, ssthresh, state, duplicate_ack_count)
                    
                    if state == FAST_RECOVERY and duplicate_ack_count >= DUP_ACK_THRESHOLD:
                        missing_seq = last_ack_received + 1
                        if missing_seq in unacked_packets:
                            packet, _ = unacked_packets[missing_seq]
                            print(f"Fast Recovery: Retransmitting packet {missing_seq}")
                            server_socket.sendto(packet, client_address)
                            unacked_packets[missing_seq] = (packet, time.time())

            except socket.timeout:
                print("Timeout occurred, retransmitting unacknowledged packets")
                cwnd, ssthresh, state = handle_timeout(cwnd, ssthresh)
                duplicate_ack_count = 0
                retransmit_unacked_packets(server_socket, client_address, unacked_packets)

def retransmit_unacked_packets(server_socket, client_address, unacked_packets):
    """Retransmit all unacknowledged packets."""
    for seq_num, (packet, _) in sorted(unacked_packets.items()):
        print(f"Retransmitting packet {seq_num}")
        server_socket.sendto(packet, client_address)
        unacked_packets[seq_num] = (packet, time.time())

def main():
    parser = argparse.ArgumentParser(description='Reliable file transfer server over UDP.')
    parser.add_argument('server_ip', help='IP address of the server')
    parser.add_argument('server_port', type=int, help='Port number of the server')

    args = parser.parse_args()
    send_file(args.server_ip, args.server_port)

if __name__ == "__main__":
    main()

