#!/usr/bin/env python3

from disorder import application
from disorder.diffuse import scattering

scattering.parallelism()

if __name__ == "__main__":
    application.run()
