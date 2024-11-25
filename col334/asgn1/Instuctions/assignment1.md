# Getting to know Network Traffic


**Goal**: The goal of this assignment is to make you familiar with network
data collection, traffic analysis, and basic network measurement tools. Appropriate
hints have been provided throughout the assignment. If you still have questions,
you are encouraged to start discussion on Piazza. 

## Network Measurement Tools:
The first part of the assignment will involve using `ping` and `traceroute`. 
You can read about these tools from : 
1. [ping](https://www.ibm.com/docs/en/aix/7.2?topic=p-ping-command)      
2. [traceroute](https://www.ibm.com/docs/sl/aix/7.1?topic=t-traceroute-command)

## Ping
Ping the following two websites: `google.com` and `sigcomm.org` (FYI, Sigcomm is the top conference in computer networking.). You should ping these websites 10 times and attach screenshots for each case. Also, perform the ping from two different networks, first from within the IITD network and second by connecting your laptop to your cellular network using hotspot. 

A. Compare the average ping latencies for the two websites in the same network. What could be the reason for differences in average ping latency? Now, compare the difference between latencies of the two networks for the same website. Explain the potential reasons for the difference. You can refer to traceroute results in the next part to answer this question. 

B. Explain the protocol being used by the ping tool. What is theoretical upper limit of packet size for the ping protocol? Are you able to ping the websites with the theortical maximum? Explain why or why not. 

C. Now try to force both the networks to ping using IPv6. Explain how you did it and were you successful (attach the relevant screenshots)? If not, what is the reason in each case?  

## Traceroute
Log the server IP addresses for the two websites. Use traceroute to find the path taken by the packets in each of the four cases (<website, network> pair) and attach the screenshot. 

A. Mention the number of IP hops as well as the list of [autonomous systems](https://en.wikipedia.org/wiki/Autonomous_system_(Internet)) observed in each case. Note there are online tools that can map an IP address to its autonomous system. 


B. Did you observe "*" in your output? If yes, explain the reason. 

C. Did you observe multiple IP address for the same hop count? If yes, explain the reason. 

D. Do you observe a 3-tiered (or a 2-tiered) Internet architecture in any traceroute? What is happening in the case where you don't observe such a architecture? 

E. Try to geolocate the IP addresses. You can use two different methods: First, try doing the reverse DNS lookup on the IP address and see if you can infer the location from the DNS address. If the reverse DNS lookup fails, use the Maxmind database for IP geolocation. Note the IP geolocation can sometimes be wrong, especially if you are using Maxmind database. In fact, accurate geolocation of IP addresses is still an active area of research. Now compare the geographical path with the observed RTTs. Do these intuitively make sense? Explain why. 

## Network Data Collection and Header analysis
For this part, you need to first collect network traffic for a 2-person 1-minute long Microsoft Teams call. Keep the video and microphone on during the call. You should do this part in a pair. You can use Wireshark or CLI tools such as tcpdump to collect the network data. Answer the following questions:

A. What are the network, transport, and application-layer protocols used by the Teams call? Log the number of packets for each protocol as percentage of total packets. Try to identify as many application-layer protocols in the traffic. [Hint: You can use wireshark filters for this analysis]

B. Do you observe a direct connection between the two hosts? If not, what is the endpoint for each host (both IP and the network)? Is it the same end-point or not? Explain what could be happening if it is not a direct connection. 

C. Identify the audio and video packets from the traffic capture and report their number. Explain the logic that you used. Plot a time-series diagram showing the bandwidth utilization by the two media types. You can use either wireshark display filters or write a script for this analysis. 

Note, you need to upload the PCAP along with the submission. 

## Network traffic analysis
In this part, you are given a network traffic trace corresponding to a speed test using M-Lab NDT7 tool. The NDT7 speed test tool works by flooding the network path between client and the server for a pre-decided duration and logs the observed throughput. This is done for both downlink (server to client) and uplink (client to server) directions. Note that there might have been some background traffic while the test was running which also gets logged in the traffic trace. You need to achieve the following objectives:

A. Isolate the traffic corresponding to the speed test from the background traffic. You will need to read about how the NDT7 speed test works. What percentage of traffic is the speedtest. 

B. Plot a time-series of observed throughput over time in each direction. You can plot average throughput per second.  

C. Find the average download and upload speeds.

You need to write a python script to answer the above question. There are python libraries like scapy and dpkt to parse a PCAP file. Your script should be named `speedtest_analysis.py`. It should take the PCAP as input as well as the following command-line arguments, each corresponding to one question:

- `--plot` should output a single time-series plot corresponding to part B
- `--throughput` should output the average download and upload speeds as comma-separated values

For example, `python speedtest_analysis.py speed.pcap --plot` should answer part B

## Submission instructions
You submission should contain a single PDF (other formats will not be graded) called report.pdf. 
- For part 1, you should attach the screenshots in the PDF itself. All questions in the first part should be answered in the report.
- For part 2, you should submit the collected PCAP. Name it <entry_no_1>_<entry_no_2>.pcap with your and your partner's entry number. The answers to the questions should be in the PDF. If you used wireshark (highly-encouraged) for the analysis, please mention the filters used as well as the steps (briefly) to get the plot. If you use a script to analyze traffic, please submit the script and name it vca.py
- For part 3, you should include the logic, plot and observed throughput values in the PDF. In addition, submit the `speedtest_analysis.py` file

Please submit a single zipped file containing all the above files
