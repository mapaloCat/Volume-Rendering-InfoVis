# Volume-Rendering-InfoVis

This repository aims at developing a volume renderer based on a raycasting approach. The skeleton code included a set-up of the whole graphics pipeline, a reader for volumetric data, and a slice-based renderer, and the main task is to extend this to a full-fledged raycaster. The implemented raycaster is then used to produce an insightful visualization of the 2013 contest data concerning the area of developmental neuroscience. The contest data of the Scientific Visualization community can be found at http://sciviscontest.ieeevis.org/2013/VisContest/index.html.  

The ultimate goal of the project is to:

* Visualize multiple volume data sets of the energy simultaneously.
* Perform volume rendering on the annotation data, which is a labeled volume data set. Each voxel has a label that is the ID of a particular neural structure.
* Combine the rendering of the energy volume data and the annotation volume data.
