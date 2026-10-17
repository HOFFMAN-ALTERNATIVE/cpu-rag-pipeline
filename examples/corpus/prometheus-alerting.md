# Prometheus alerting

## Writing alert rules
Alert on symptoms users feel, such as error rate and latency, rather than on every cause.
Use a `for:` duration so short spikes do not page anyone.

## Routing with Alertmanager
Alertmanager groups, deduplicates and routes alerts to receivers like email, chat or a
pager. Inhibition rules silence dependent alerts when a whole host is down.

## Testing rules
promtool can unit-test alerting rules against synthetic time series, which catches typos
in label matchers before they reach production.
