This program calculates trade routes using the rules in GURPS Traveller: Far
Trader.

https://github.com/makhidkarun/traveller_pyroute already exists and does a lot
more, but I couldn't get it to work, so I'm writing my own.


Run "python3 traderoutes.py -h" for help.

Map data is downloaded from http://travellermap.com


Runtime dependencies:

Python 3.8 or later
http://travellermap.com if you haven't already downloaded data locally
scipy for floyd_warshall
numpy for scipy
pycairo for PDF drawing


Development dependencies:

pytest for running unit tests
mypy for type checking
black for code formatting
git and github for version control
