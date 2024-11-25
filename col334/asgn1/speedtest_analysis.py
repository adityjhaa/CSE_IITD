import argparse
from scapy.all import *
import matplotlib.pyplot as plt
from collections import defaultdict

def parse_pcap(pcap_file):
    return rdpcap(pcap_file)

def find_port(packets):
    connection = {}

    for packet in packets:
        if packet.haslayer(IP) and packet.haslayer(TCP):
            sport = packet[TCP].sport
            dport = packet[TCP].dport
            
            if sport not in connection:
                connection[sport] = 0
            if dport not in connection:
                connection[dport] = 0
                
            connection[sport] += 1
            connection[dport] += 1

    server_port = max(connection, key=connection.get)

    return server_port

def isolate_speedtest_traffic(packets, port = 443):
    return [pkt for pkt in packets if TCP in pkt and (pkt[TCP].sport == port or pkt[TCP].dport == port)]


def calculate_throughput(packets, port = 443):
    download_data = defaultdict(int)
    upload_data = defaultdict(int)
    
    for pkt in packets:
        if TCP in pkt:
            timestamp = pkt.time
            size = len(pkt[TCP].payload)
            if pkt[TCP].dport == port:
                download_data[int(timestamp)] += size
            elif pkt[TCP].sport == port:
                upload_data[int(timestamp)] += size
    
    return download_data, upload_data

def plot_throughput(download_data, upload_data):
    plt.figure(figsize=(12, 8))
    plt.plot(list(download_data.keys()), [v*8/1000000 for v in download_data.values()], label='Download')
    plt.plot(list(upload_data.keys()), [v*8/1000000 for v in upload_data.values()], label='Upload')
    plt.xlabel('Time (s)')
    plt.ylabel('Throughput (Mbps)')
    plt.title('NDT7 Speed Test Throughput')
    plt.legend()
    plt.grid(True)
    plt.savefig('throughput_plot.png')
    plt.close()


def calculate_average_speeds(download_data, upload_data):
    total_download = sum(download_data.values()) * 8 / 1000000
    total_upload = sum(upload_data.values()) * 8 / 1000000
    duration = max(max(download_data.keys()), max(upload_data.keys())) - \
               min(min(download_data.keys()), min(upload_data.keys()))
    
    avg_download = total_download / duration if duration > 0 else 0
    avg_upload = total_upload / duration if duration > 0 else 0
    
    return avg_download, avg_upload

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pcap_file", help="Path to the PCAP file")
    parser.add_argument("--plot", action="store_true", help="Generate throughput plot")
    parser.add_argument("--throughput", action="store_true", help="Output average speeds")
    args = parser.parse_args()

    packets = parse_pcap(args.pcap_file)
    port = find_port(packets)
    speedtest_packets = isolate_speedtest_traffic(packets, port)

    total_traffic = sum(len(pkt) for pkt in packets)
    speedtest_traffic = sum(len(pkt) for pkt in speedtest_packets)

    percentage = (speedtest_traffic / total_traffic) * 100

    print(f"Speed test traffic: {percentage:.2f}%")

    if args.plot:
        download_data, upload_data = calculate_throughput(speedtest_packets, port)
        plot_throughput(download_data, upload_data)

    if args.throughput:
        download_data, upload_data = calculate_throughput(speedtest_packets, port)
        avg_down, avg_up = calculate_average_speeds(download_data, upload_data)
        print(f"{avg_down:.2f},{avg_up:.2f}")


if __name__ == "__main__":
    main()

