# SSH hardening

## Key-based authentication
Disable password logins and rely on Ed25519 keys. Set `PasswordAuthentication no` and
`KbdInteractiveAuthentication no` in sshd_config, then reload the daemon. Keys should be
protected with a passphrase and loaded through an agent.

## Restricting root
Set `PermitRootLogin no` so administrators log in as themselves and escalate with sudo.
This leaves an audit trail of who did what.

## Rate limiting brute force
Tools like fail2ban watch the auth log and ban addresses after repeated failed attempts.
Combine this with a firewall rule that limits new connections per minute to port 22.
