This program calculates trade routes using the rules in GURPS Traveller: Far
Trader.

https://github.com/makhidkarun/traveller_pyroute already exists and does a lot
more, but I couldn't get it to work, so I'm writing my own.


Run "python3 traderoutes.py -h" for help.

Map data is downloaded from http://travellermap.com


Dependencies for use:

Python 3.8 or later
http://travellermap.com if you haven't already downloaded data locally


Dependencies for development:

pytest for running tests
mypy for type checking
black for code formatting
git and github for version control
scipy for floyd_warshall
numpy for scipy


TODO:
What library should we use to write PDFs from Python and possibly Rust?
Contenders appear to be skia, pdfium, and cairo.
