# Network Module

This document will serve as the design guideline for the implementation of the
network module targeted for PySAL 1.7 release (January 31, 2014).

## Origins

The network module will implement the feature set from [GeoDaNet][GeoDaNet]

## Structure

### Analytics (s)

PySAL network will include the following spatial analytical methods

 - Nearest Neighbor Distance Distributions
 - Global Network Autocorrelation
 - Global K-Functions
 - Local Indicators of Network-Constrained Clusters (LINCS)
 - Local K-Functions
 - Network Kernels
 - Accessibility Indices

### Utility (j)

 - Network Edge Segmentation
 - Assignment of Counts/Rates to segments, nodes, edges
 - ~~Network Edge Length~~
 - ~~Snapping off-network objects to network~~ (Only internal observations)
 - ~~Single shortest path (Dijkstra's) (One node and all other nodes)~~
 - ~~Shortest Path (Path between two nodes)~~
 - ~~Network Connectivity~~
 - Extended shortest path
 - Network Voronoi Diagrams ?
 - Simulated points on a network
     - ~~Uniform~~
     - Nonuniform
 - ~~Threshold distance~~
 - Network Center
 - Intersection with buffering (s)
 - Node insertion / deletion

### Data Structures (j)

 - ~~Winged Edge Data Structure~~
 - ~~Extraction of WED from a planar polyine shapefile~~
 - Spatial Weights for Networks
     - Link Contiguity
         - ~~First Order~~
     - Distance Based
         - KNN
         - Threshold   
 - Handle direction

### FileIO

#### Input
 - ~~reading polyline shapefiles~~
 - point shapefiles
 - polygon files
 - wed
 - dot
 - (geo)json
 - binary

#### Output
 - wed
 - dot
 - shapefiles
 	- polyline
	- points
	- polygons
 - (geo)json
 - binary

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

- holes in the wed


[GeoDaNet]: https://geodacenter.asu.edu/drupal_files/Geodanet_Manual_03_2012.pdf
