This program calculates trade routes using the rules in GURPS Traveller: Far
Trader.

https://github.com/makhidkarun/traveller_pyroute already exists, but I
couldn't get it to work, so I'm writing my own.

Note that I have ported this code to Rust.  The new version is at
https://github.com/dripton/traderust  It's *much* faster, and also easier
to install.  But I'll keep the Python version up in case anyone wants it.

Run "python3 traderoutes.py -h" for help.

Map data is downloaded from http://travellermap.com


Runtime dependencies:

* Python 3.9 or later
* http://travellermap.com if you haven't already downloaded data locally
* scipy for Dijkstra all pairs shortest paths
* numpy for fast 2D arrays
* pycairo for PDF drawing
* apsp_mt (optional) for much faster Dijkstra all pairs shortest paths


Development dependencies:

* pytest for running unit tests
* mypy for type checking
* black for code formatting
* git and GitHub for version control
