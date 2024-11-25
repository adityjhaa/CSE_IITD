import socket
import argparse
import json

MSS = 1400

def receive_file(server_ip, server_port):
    """Receive the file from the server"""
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client_socket.settimeout(2)

    server_address = (server_ip, server_port)
    expected_seq = 0
    output_file_path = "received_file.txt"
    received_buffer = {}

    with open(output_file_path, 'wb') as file:
        connection_attempts = 0
        max_attempts = 5
        connected = False

        while not connected and connection_attempts < max_attempts:
            try:
                print("Sending START signal to server")
                client_socket.sendto(b"START", server_address)
                send_ack(client_socket, server_address, 0)
                connected = True
            except socket.timeout:
                connection_attempts += 1
                print(f"Connection attempt {connection_attempts} failed, retrying...")
        
        if not connected:
            print("Failed to connect to server")
            return

        while True:
            try:
                packet, _ = client_socket.recvfrom(MSS + 1024)
                packet_data = json.loads(packet.decode())
                
                if packet_data['fin']:
                    print("Received END signal from server")
                    send_ack(client_socket, server_address, -1)
                    print("Sent ACK for END signal to server")
                    break

                seq_num = packet_data['seq']
                size = packet_data['size']
                data = packet_data['data'].encode('latin1')[:size]

                if seq_num == expected_seq:
                    file.write(data)
                    expected_seq += 1

                    while expected_seq in received_buffer:
                        file.write(received_buffer[expected_seq])
                        del received_buffer[expected_seq]
                        expected_seq += 1

                    send_ack(client_socket, server_address, expected_seq)

                elif seq_num > expected_seq:
                    print(f"Received out-of-order packet {seq_num}, buffering")
                    received_buffer[seq_num] = data
                    send_ack(client_socket, server_address, expected_seq)

            except socket.timeout:
                print("Timeout waiting for data")
                send_ack(client_socket, server_address, expected_seq)
            except json.JSONDecodeError:
                print("Received malformed packet, ignoring")

def send_ack(client_socket, server_address, seq_num):
    """Send ACK k to indicate packet k-1 has been received"""
    ack_packet = json.dumps({'ack': seq_num}).encode()
    client_socket.sendto(ack_packet, server_address)

def main():
    parser = argparse.ArgumentParser(description='Reliable file receiver over UDP.')
    parser.add_argument('server_ip', help='IP address of the server')
    parser.add_argument('server_port', type=int, help='Port number of the server')

    args = parser.parse_args()
    receive_file(args.server_ip, args.server_port)

if __name__ == "__main__":
    main()

