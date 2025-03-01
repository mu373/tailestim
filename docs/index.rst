tailestim
==================================

tailestim is a Python package for estimating tail parameters of heavy-tailed distributions, which is useful for analyzing power-law behavior in complex networks.

.. note::
   The original estimation implementations are from `ivanvoitalov/tail-estimation <https://github.com/ivanvoitalov/tail-estimation>`_, which is based on the paper `"Scale-free networks well done" <https://doi.org/10.1103/PhysRevResearch.1.033034>`_ (Voitalov et al. 2019). tailestim is a wrapper package that provides a more convenient/modern interface and logging, that can be installed using ``pip`` and ``conda``.

Features
--------

- Multiple estimation methods including Hill, Moments, Kernel, Pickands, and Smooth Hill estimators
- Double-bootstrap procedure for optimal threshold selection
- Built-in example datasets

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   usage

.. toctree::
   :maxdepth: 2
   :caption: API Documentation

   api

References
---------

- I. Voitalov, P. van der Hoorn, R. van der Hofstad, and D. Krioukov. Scale-free networks well done. *Phys. Rev. Res.*, Oct. 2019, doi: `10.1103/PhysRevResearch.1.033034 <https://doi.org/10.1103/PhysRevResearch.1.033034>`_.
- I. Voitalov. `ivanvoitalov/tail-estimation <https://github.com/ivanvoitalov/tail-estimation>`_, GitHub. Mar. 2018.

Index
---------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
