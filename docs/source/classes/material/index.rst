Material
========

.. toctree::
   :glob:
   :maxdepth: 1

   *

The classes in the `material` package.
Using an instance of the `UnitCell` class from `CIF` file, its methods provide direct access to the properties of the crystallographic unit cell.
As an example, the structure factors can be calculated using a function from the `structure` module.

.. plot:: classes/material/structure.py
   :include-source:
   :caption: Structure factors.

.. csv-table:: Cu3Au
   :file: Cu3Au.csv
   :header: "h", "k", "l", "d", "F(real)", "F(imag)", "Mult"
   :widths: auto
