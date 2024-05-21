```
usage: test.py [-h] (-a | -l0 | -lc | -l1) [-noclear]

options:
  -h, --help  show this help message and exit
  -a          all
  -l0         l0: sanity check, finish in less than 1 minute.
  -lc         lc: cross check. Check options that do not affect generated results.
  -l1         l1: fast check, finish in less than 10 minutes.
  -noclear    do not clear the output directory. Useful for skipping some tests.
```
