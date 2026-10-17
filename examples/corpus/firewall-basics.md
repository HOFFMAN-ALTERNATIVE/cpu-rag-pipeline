# Host firewall basics

## Default deny
Start from a policy that drops all inbound traffic, then open only the ports a service
needs, such as 443 for HTTPS. Outbound traffic can usually stay allowed.

## nftables and ufw
nftables is the modern Linux packet filtering framework; ufw is a friendly front end that
generates rules for common cases.

## Docker caveat
Docker inserts its own iptables rules for published ports, which can bypass ufw. Bind
published ports to 127.0.0.1 when a reverse proxy sits in front.
