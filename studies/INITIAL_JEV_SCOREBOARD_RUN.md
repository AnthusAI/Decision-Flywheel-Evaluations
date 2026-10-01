# Initial Jev scoreboard run

Status: live approval granted by the human owner on 2026-10-01 for the two
scoreboard collections below, at most 60,900 physical attempts per dataset.
It authorizes no ordering or cross-model calls. No scoreboard results are
reported here. Development results below are selection evidence only and are
not performance claims.

## Development outcome

All development collections completed under
[INITIAL_JEV_DEVELOPMENT_RUN.md](INITIAL_JEV_DEVELOPMENT_RUN.md): AG News
8,000/8,000 and Emotion 12,000/12,000 requests succeeded (Emotion needed one
retry). Each dataset's 20 random-balanced trials completed before a winner was
frozen from development labels only.

- AG News: winner `random-balanced:d26cd1c97b1c:16` (seed 1), development accuracy 0.92.
- Emotion: winner `random-balanced:3981e3e02c57:64` (seed 4), development macro-F1 0.6887.

Text-free artifacts, derived protocols, and registrations are in
[selected/](selected/).

## Frozen scoreboard identities

AG News (33 conditions × 2,000 targets = 66,000 logical; 58,000 physical):

- Derived protocol: `0daf01d0eaf417b7a2438cd00898fb31b78634d25aab95d4c96e4570e44acf2d`
- Scoreboard preflight: `6a65806853cf3c77781a868e76ddd5b52eb9789b494cc0b360725b02732715f0`
- Approved cumulative ceiling: 60,900 physical attempts

Emotion (33 conditions × 2,000 targets = 66,000 logical; 58,000 physical):

- Derived protocol: `3b9bdde0a072f857b8ccb30bea369f3fb4e23bb724d6c1083546bf525b74083a`
- Scoreboard preflight: `814b6596e1dd7d96c629fadaa45a5e8b20d5b1a810871803f8e8cf764a11d16a`
- Approved cumulative ceiling: 60,900 physical attempts

The preflights are large and regenerable from the committed artifacts:
`cli preflight --protocol selected/<dataset>.protocol.json --stage scoreboard`.
Provider, transport, retry policy (at most one explicit outer retry per failed
request within the ceiling, SDK retries off), and limits match the development run.
Separate ledger per dataset. The scoreboard is not used to tune anything; the
analysis is the descriptive paired design in INITIAL_JEV_STUDY.md.
