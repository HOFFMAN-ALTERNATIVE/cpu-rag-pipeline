# TLS certificates

## Automatic issuance
The ACME protocol lets a server prove control of a domain and obtain a certificate from
Let's Encrypt without manual steps. Clients renew certificates well before the 90 day expiry.

## Modern protocol settings
Allow only TLS 1.2 and TLS 1.3, prefer ECDHE key exchange and AEAD ciphers, and enable
HSTS once every subdomain serves HTTPS.

## Monitoring expiry
Alert when a certificate expires in fewer than 14 days. A forgotten renewal is one of the
most common causes of avoidable outages.
