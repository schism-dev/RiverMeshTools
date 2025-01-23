# RiverMapper: DEM-based meshing aid for compound flooding 
![Watershed rivers](Intro.jpg?raw=true)

Note: 
* This is not a mesh generator; rather, it generates river arcs or polygons that can be used as input for mesh generators. The primary goal of the tool is to eliminate the manual effort of delineating watershed rivers, particularly for continental-scale compound flood studies.

* The tool operates independently of RiverMeshTools/pyDEM. The required 1D river network can be any reasonable approximation of river positions, including:
  - thalwegs extracted by RiverMeshTools/pyDEM or similar packages, as well as tools such as ArcGIS and QGIS with similar functions.
  - existing hydrological products such as flowlines or streamlines from [National Hydrography Dataset](https://www.epa.gov/waterdata/nhdplus-national-hydrography-dataset-plus) and [National Water Model](https://water.noaa.gov/about/nwm).
  - hand-drawn flow lines for smaller-scale applications.
   
* The positions of the 1D streams do not need to align precisely with the thalweg; the tool utilizes DEMs (Digital Elevation Models) to accurately locate the thalweg positions within a search range determined by the local river width.

* As a special use case, RiverMapper can leverage the "Area" polygons from the [National Hydrography Dataset](https://www.epa.gov/waterdata/nhdplus-national-hydrography-dataset-plus) instead of DEMs to generate river arcs. This approach often provides a cleaner delineation of small river channels, particularly those inadequately represented in the DEMs. 
The figure below shows an example near the Pearl River, Louisiana:
![nhd-guided-meshing](https://github.com/user-attachments/assets/df449d40-80b0-49d2-998f-743e73923fd9)
This method is being tested with the latest developmental version of [STOFS-3D-Atlantic](https://registry.opendata.aws/noaa-nos-stofs3d/) and a sample application will be added soon.


## Installation 
Python 3.9 or above is recommended

```bash
pip install git+https://github.com/schism-dev/RiverMeshTools.git#subdirectory=RiverMapper
```

You may need to manually install gdal and/or mpi4py (e.g., conda install gdal or mamba install gdal) if there is an error related to it, then redo the above 'pip install' command.

## Sample applications
Download link:
http://ccrm.vims.edu/yinglong/feiye/Public/RiverMapper_Samples.tar

Use the command
```bash
tar xvf RiverMapper_Samples.tar
```
to extract files from RiverMapper_Samples.tar.

## Tutorial
See online documentation in the [SCHISM manual](https://schism-dev.github.io/schism/master/mesh-generation/meshing-for-compound-floods/generate-river-map.html) for more information.
