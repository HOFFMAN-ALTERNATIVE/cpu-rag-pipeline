# Backups that actually restore

## The 3-2-1 rule
Keep three copies of your data on two different media with one copy off-site. Object
storage in another region is a convenient off-site target.

## Encryption and deduplication
Tools such as restic and borg encrypt data client-side and deduplicate blocks, so daily
snapshots cost little extra space.

## Restore drills
A backup is only proven by restoring it. Schedule automated restore tests into a scratch
environment and verify checksums of the recovered files.
