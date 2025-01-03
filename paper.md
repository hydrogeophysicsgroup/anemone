```
---
title: 'anEMone: A Python package for modelling electromagnetics over 1-dimensional earth'
tags:
  - python
  - pytorch
  - geophysics
  - electromagnetics

authors:
  - name: Matthew P. Griffiths
    orcid: 0000-0002-5674-539X
    corresponding: true
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Egil Möller
    affiliation: 2
  - name: Ivan Vela
    affiliation: 1
  - name: Denys Grombacher
    affiliation: 1
  - name: Ander Vest Christiansen
  - affiliation: 1

affiliations:
 - name: Hydrogeophysics Group, Dept. of Geoscience, Aarhus University, Denmark
   index: 1
 - name: EMerald Geomodelling, Norway
   index: 2
date: 01 January 2025
bibliography: paper.bib
---

```

# Summary

Anemone is a simulation software for geophysical electromagnetic (EM) measurements. 
Simulations are restricted to horizontally layered earth scenarios; however complex conductivity effects can be accommodated.
This python package is built with Pytorch to leverage GPUs for highly parallel simulations.
Furthermore the code itself is designed to closely mathematical formulations of the physical theory. 
References to equations in foundational literature have been included in the documentation of the functions.


# Statement of need

Electromagnetic methods are crucial tools for imaging near-surface geological structures. 
Simulating these measurements are key to interpretting the underlying earth structure. 
While there are several software solutions available for modelling electromagnetics [@auken2015], including open source Python based packages [@crockett2017], these codes tend to be buried in massive projects, bundled with simulations codes of other geophysical methods.
This can make learning curve relatively steep. 
Anemone is built with a focus on surface-based EM for 1-dimensional earth models. The objects in the code are designed specifically for EM to be intuitive for practicing EM geophysicists.

As much as possible, the code structure is designed to resemble the mathematics formulations / notation, following that of @ward1988.
This makes it easier to read for those well verse in the EM theory, but can also serve as a learning tool for those more familiar with programming than the calculus based formulations in the literature.

Finally, the package is implemented using Pytorch, allowing for high-parallelizability of forwards leveraging GPUs.
While other packages have implemented CPU based parallelization, this is the first to run on GPUs as well.

# Acknowledgements

