# Network Module

This document will serve as the design guideline for the implementation of the
network module targeted for PySAL 1.7 release (January 31, 2014).

## Origins

The network module will implement the feature set from [GeoDaNet][GeoDaNet]

## Structure

### Analytics

PySAL network will include the following spatial analytical methods

 - Global Network Autocorrelation
 - Global K-Functions
 - Local Indicators of Network-Constrained Clusters (LINCS)
 - Local K-Functions
 - Network Kernels
 - Network Voronoi Diagrams ?

### Utility

 - Network Edge Segmentation
 - Assignment of Counts/Rates to segments, nodes, edges
 - Network Edge Length
 - Snapping off-network objects to network

### Data Structures

 - Winged Edge Data Structure
 - Extraction of WED from a planar polyine shapefile
 - Spatial Weights for Networks


### Module Listing

List key modules here and their purposes


 - `shp2graph.py` ?
 - `wed.py` core winged edge data structure
 - `util.py` utility functions

### Notebooks

List notebooks and their purpose

## Development Plans

Prioritize components for 1.7 development


## Issues




[GeoDaNet]: https://geodacenter.asu.edu/drupal_files/Geodanet_Manual_03_2012.pdf
