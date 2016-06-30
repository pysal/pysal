OGC Spec boildown
=================
A shape is *simple* if it is not self-intersecting. A shape is closed when it
has a nonempty interior.  

A **Point** is a zero-dimensional set of coordinates {X,Y[,Z,M}}, where X,Y
represent a mandatory two-dimensional position vector, Z is an optional
third dimension position, and M is an optional mark/measure. Its boundary
is empty, and its interior is the point itself. It is always *simple*. 

A **MultiPoint** is a zero-dimensional set of Points. It is *simple* when no
two **Points** are equal. Its boundary is empty. 

A **LineString** is a curve with linear interpolation between its endpoints. It
is *simple* if it has no self-intersection. 

A **Line** is a **LineString** with exactly 2 **Points**. It is *simple*. 

A **LinearRing** is a **LineString** that is both *closed* and *simple*. 

A **MultiLineString** is a collection of LineStrings. It is *simple* iff all of
its elements are simple AND self-intersection only occurs at the boundary of its
elements. The boundary of a **MultiLineString** is defined as the set of Points
on the boundary of an *even* number of the constituent **LineString**s. 

A **Polygon** is a surface defined by 1 exterior **Linear Ring** and 0 or more
interior boundaries, called "holes." The "top" of the shape is the space bound
to the left of the exterior **Linear Ring** when traversed in a
counter-clockwise direction. In addition, the top is on the left of holes when
traversed on the clockwise direction. They are topologically closed. Their
boundary consists of the set of the **Linear Rings** composing the polygon.
**Linear Rings** in the boundary can share tangent points, but otherwise cannot
intersect. Since a \ cl(a) = int(a), it must be the case that (cl(int(a)) = a).
So, no cutlines/spikes can exist in the shape. 

A **MultiPolygon** is a collection of **Polygons**. For all of its constituent
**Polygons**, no interior may intersect any other interior. In addition,
boundaries may only intersect at a finite number of points. Since it is a
collection of **Polygons**, no cutlines/spikes are allowed. 


Section 6.1.15.3, Spatial Relationship details
-----------------------------------------------
consider two valid OGC geometries a,b as closed point sets. Then, define 
the standard Egenhoffer operators:

closure(a) = boundary(a) = cl(a), the subset of points {x in a} such that {x +
v not in a} infinitesimal perturbation vector v. Thus, without a \
cl(a) is an open set. 

interior(a) = int(a), the subset of points {y in a} such that there exists no
infinitesimal v such that {y + v not in a}. a \ cl(a) = int(a), and int(a) is an
open set. 

Operators/Relational predicates will have first words capitalized. Lowercase
statements are to be interpreted as set operators. Thus, Intersects refers to
the spatial predicate and intersects refers to point set intersection. Likewise,
Equals refers to topological equality defined below, but = refers to
mathematical equality, in the sense of a number being equal to another number.

Logical junctions/disjunctions/negations will be allcaps. The word "empty"
refers to the empty set, and is a set property. 

Equals
--------
- "a Equals b" iff (a contains b) and (b contains a)
- Note that this means that a and b's vertex set can be different.

Disjoint
---------
- "a Disjoint b" iff (a intersect b) is empty. 

Touches
---------
- "a Touches b" iff (int(a) intersect int(b)) is empty AND (a intersect b) is NOT empty

Crosses
-------
- "a Crosses b" iff (int(a) intersect int(b) is NOT empty) AND (a intersect b
  NOT Equals a) AND (b intersect a NOT Equals b)

Within
-------
- "a Within b" iff (a intersect b Equals a) AND (int(a) intersect cl(b) is
  empty) 

Overlaps
--------
- For this, define `dim(X)` as the operator returning the topological dimension
  of `X`
- Overlaps is only defined for a and b such that dim(A) == dim(b):
- "a Overlaps b" iff (dim(int(a)) = dim(int(b)) = dim(int(a) intersect int(b)))
  AND (a intersect b NOT Equals a) AND (a intersect b NOT Equals b)
- in plain language, this occurs when two shapes intersect over part of their
  interior and that intersection has the same dimensionality as the intersecting
  shapes. 

Contains
--------
- "a Contains b" iff (b Within a)

Intersects
----------
- "a Intersects b" iff NOT (a Disjoint b)
