"""Prepare NHD Area polygons for use as RiverMapper surrogate DEMs.

The command-line workflow splits a polygon shapefile into overlapping tiles,
rasterizes polygon interiors as ``-1`` (water) with ``0`` background (land),
and creates a coarse, globally covering ``global_dummy.tif`` by default.

Basic usage::

    python Scripts/pre_proc_nhd_area.py nhdarea.shp

Choose an output directory or override the rasterization defaults::

    python Scripts/pre_proc_nhd_area.py nhdarea.shp \
        --outdir nhdarea_tifs --tile-size 0.5 --pixel-size 2e-5

Pass the generated high-resolution tiles to ``make_river_map`` before the
dummy TIF so the first-available-tile priority preserves water values::

    make_river_map(
        tif_fnames=[*nhd_area_tiles, global_dummy_tif],
        nhd_area_tif=True,
        ...,
    )

This module also provides :func:`build_clean_polygon_mask` for optional
geometry cleanup before rasterization. Cleanup is intentionally separate from
the CLI rasterization step: it reprojects to a metric CRS, repairs and
dissolves polygons, and returns a GeoDataFrame for the caller to inspect and
save. Reproject a cleaned mask back to EPSG:4326 before using the default CLI
tile and pixel sizes, which are expressed in degrees.
"""

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from pyproj import CRS
from pyproj.exceptions import CRSError
from shapely import Polygon


PRECISION = 3  # decimal places used when snapping coordinates in the target CRS
GAP = 0.1  # meters; morphological-closing distance for gaps and slivers


def build_clean_polygon_mask(poly_shp, target_crs="esri:102008"):
    """Build one repaired polygon mask without overlaps or narrow gaps.

    The input is reprojected to ``target_crs``, snapped to a precision grid
    when Shapely supports it, repaired, dissolved, and morphologically closed
    with :data:`GAP`. The returned one-row GeoDataFrame remains in the metric
    target CRS; callers should inspect it and choose the desired output CRS.
    ``target_crs`` should be a suitable local or regional projected CRS with
    meter units. It defaults to ESRI:102008 for North America. A
    geographic CRS or a projected CRS using feet or other units raises
    ``ValueError``.

    Similar cleanup can be performed in QGIS with "Delete duplicate
    Geometries" followed by positive and negative buffers, which may be faster
    for very large inputs.
    """
    try:
        target_crs = CRS.from_user_input(target_crs)
    except CRSError as exc:
        raise ValueError(f'Invalid target_crs: {target_crs}') from exc

    horizontal_axes = target_crs.axis_info[:2]
    uses_meters = (
        target_crs.is_projected and len(horizontal_axes) == 2 and
        all(np.isclose(axis.unit_conversion_factor, 1.0) for axis in horizontal_axes)
    )
    if not uses_meters:
        units = ', '.join(axis.unit_name or 'unknown' for axis in horizontal_axes)
        raise ValueError(
            f'target_crs must be a projected CRS with meter units; '
            f'{target_crs.to_string()} uses {units or "unknown units"}'
        )

    polys = gpd.read_file(poly_shp)

    # Reproject to a metric CRS so GAP is in meters.
    polys = polys.to_crs(target_crs)

    # Snap coordinates to a small grid to collapse near-coincident edges.
    try:
        from shapely import set_precision  # shapely>=2.0
        polys["geometry"] = polys.geometry.apply(lambda g: set_precision(g, PRECISION))
    except Exception:
        pass

    # Repair, dissolve overlaps, close narrow gaps/slivers, and repair again.
    polys["geometry"] = polys.buffer(0)
    mask = polys.dissolve().geometry.iloc[0]
    mask_closed = gpd.GeoSeries(mask, crs=polys.crs).buffer(+GAP).buffer(-GAP).iloc[0]
    mask_closed = gpd.GeoSeries(mask_closed, crs=polys.crs).buffer(0).iloc[0]

    return gpd.GeoDataFrame(geometry=[mask_closed], crs=polys.crs)


def gen_splitter(dem_box, dl, overlap_ratio=0.01, crs='EPSG:4326'):
    """Generate overlapping square tile bounds covering ``dem_box``.

    Args:
        dem_box: Bounding box ``[xmin, ymin, xmax, ymax]``.
        dl: Side length of each square in CRS units.
        overlap_ratio: Fraction of ``dl`` added around every tile.
        crs: CRS assigned to the returned splitter GeoDataFrame.

    Returns:
        A pair ``(bounds, splitter_gdf)`` containing numeric tile bounds and
        their polygon representation.
    """
    splitter = []
    for x in np.arange(dem_box[0], dem_box[2], dl):
        for y in np.arange(dem_box[1], dem_box[3], dl):
            splitter.append([
                x - dl * overlap_ratio, y - dl * overlap_ratio,
                x + dl + dl * overlap_ratio, y + dl + dl * overlap_ratio
            ])

    splitter_gdf = gpd.GeoDataFrame(
        geometry=[
            Polygon([(box[0], box[1]), (box[2], box[1]),
                     (box[2], box[3]), (box[0], box[3])])
            for box in splitter
        ],
        crs=crs,
    )
    return splitter, splitter_gdf


def split_vector_shp(shp_fname, splitter_gdf, outdir='./split/'):
    """Clip a vector shapefile into every non-empty splitter tile.

    Returns:
        list[pathlib.Path]: Paths to the generated tile shapefiles.
    """
    shp_fname = Path(shp_fname)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    input_shp_gdf = gpd.read_file(shp_fname)
    split_shp_fnames = []
    for i, splitter in enumerate(splitter_gdf.geometry):
        clipped_gdf = gpd.clip(input_shp_gdf, splitter)
        if clipped_gdf.empty:
            print(f'skip empty tile {i}')
            continue
        output_path = outdir / f'{shp_fname.stem}_{i}.shp'
        clipped_gdf.to_file(output_path)
        split_shp_fnames.append(output_path)

    return split_shp_fnames


def rasterize_shp(shp_fname, burn_value=-1, pixel_size=2e-5):
    """Rasterize one polygon shapefile into a single-band GeoTIFF.

    Polygon interiors receive ``burn_value`` and all other cells inside the
    shapefile bounds receive 0. ``pixel_size`` is expressed in the shapefile's
    CRS units; the default NHD workflow expects EPSG:4326 degrees. The output
    is written beside the shapefile with a ``.tif`` suffix.

    Returns:
        pathlib.Path: Path to the generated GeoTIFF.
    """
    shp_fname = Path(shp_fname)
    gdf = gpd.read_file(shp_fname)

    minx, miny, maxx, maxy = gdf.total_bounds
    width = max(1, int(np.ceil((maxx - minx) / pixel_size)))
    height = max(1, int(np.ceil((maxy - miny) / pixel_size)))
    transform = from_origin(minx, maxy, pixel_size, pixel_size)

    shapes = [(geom, burn_value) for geom in gdf.geometry]
    raster = rasterize(
        shapes, out_shape=(height, width), transform=transform,
        fill=0, dtype='int16')

    output_path = shp_fname.with_suffix('.tif')
    with rasterio.open(
        output_path, 'w', driver='GTiff', height=height, width=width,
        count=1, dtype='int16', crs=gdf.crs, transform=transform,
    ) as dst:
        dst.write(raster, 1)

    return output_path


def create_dummy_tif(output_path, bounds, pixel_size=1.0, value=0, dtype='uint8'):
    """Create a constant-value EPSG:4326 GeoTIFF covering ``bounds``.

    The NHD workflow uses a coarse global TIF with value 0 to provide land
    coverage wherever no high-resolution polygon tile is present.

    Returns:
        pathlib.Path: Path to the generated GeoTIFF.
    """
    output_path = Path(output_path)
    min_lon, min_lat, max_lon, max_lat = bounds
    width = int(np.ceil((max_lon - min_lon) / pixel_size))
    height = int(np.ceil((max_lat - min_lat) / pixel_size))
    transform = from_origin(min_lon, max_lat, pixel_size, pixel_size)
    data = np.full((height, width), value, dtype=dtype)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output_path, 'w', driver='GTiff', height=height, width=width,
        count=1, dtype=dtype, crs='EPSG:4326', transform=transform,
    ) as dst:
        dst.write(data, 1)

    return output_path


def rasterize_polygon_tiles(
    input_shp_fname, outdir=None, tile_size=0.5, overlap_ratio=0.01,
    pixel_size=2e-5, burn_value=-1
):
    """Split a polygon shapefile and rasterize every non-empty tile.

    A regular splitter is built over the input bounds. Each clipped shapefile
    is rasterized using ``burn_value`` for polygon interiors and 0 for its
    background. For EPSG:4326 inputs, tile and pixel sizes are in degrees.

    The default output directory is
    ``<input parent>/<input stem>_split_rasterized``.

    Returns:
        list[pathlib.Path]: Generated high-resolution surrogate GeoTIFF paths.
    """
    input_shp_fname = Path(input_shp_fname)
    if outdir is None:
        outdir = input_shp_fname.parent / f'{input_shp_fname.stem}_split_rasterized'
    else:
        outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    input_gdf = gpd.read_file(input_shp_fname)
    _, splitter_gdf = gen_splitter(
        input_gdf.total_bounds, dl=tile_size,
        overlap_ratio=overlap_ratio, crs=input_gdf.crs)

    splitter_gdf.to_file(outdir / 'splitter.shp')
    split_shps = split_vector_shp(input_shp_fname, splitter_gdf, outdir)
    return [
        rasterize_shp(tile_shp, burn_value=burn_value, pixel_size=pixel_size)
        for tile_shp in split_shps
    ]


def _positive_float(value):
    """Argparse converter requiring a finite number greater than zero."""
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError('must be a finite number greater than zero')
    return value


def _overlap_ratio(value):
    """Argparse converter requiring a fractional overlap in [0, 1)."""
    value = float(value)
    if not np.isfinite(value) or not 0 <= value < 1:
        raise argparse.ArgumentTypeError('must be a finite number in [0, 1)')
    return value


def build_arg_parser():
    """Build the CLI parser for polygon-to-surrogate-TIF preprocessing."""
    parser = argparse.ArgumentParser(
        description=(
            'Rasterize an NHD Area polygon shapefile into tiled RiverMapper '
            'surrogate DEMs (water=-1, background=0 by default).'
        ),
        epilog=(
            'Example: python Scripts/pre_proc_nhd_area.py nhdarea.shp '
            '--outdir nhdarea_tifs'
        ),
    )
    parser.add_argument('input_shp', type=Path, help='input polygon shapefile')
    parser.add_argument(
        '--outdir', type=Path,
        help='output directory (default: <input stem>_split_rasterized beside input)')
    parser.add_argument(
        '--tile-size', type=_positive_float, default=0.5,
        help='splitter tile side length in input CRS units (default: 0.5)')
    parser.add_argument(
        '--overlap-ratio', type=_overlap_ratio, default=0.01,
        help='fractional overlap added around each tile (default: 0.01)')
    parser.add_argument(
        '--pixel-size', type=_positive_float, default=2e-5,
        help='output pixel size in input CRS units (default: 2e-5)')
    parser.add_argument(
        '--burn-value', type=int, default=-1,
        help='value assigned to polygon interiors/water (default: -1)')
    parser.add_argument(
        '--no-dummy', action='store_true',
        help='do not create the global zero-valued background TIF')
    parser.add_argument(
        '--dummy-pixel-size', type=_positive_float, default=1.0,
        help='global dummy TIF pixel size in degrees (default: 1.0)')
    return parser


def main(argv=None):
    """Run polygon tiling and rasterization from command-line arguments."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    input_shp = args.input_shp.expanduser()
    if not input_shp.is_file():
        parser.error(f'input shapefile does not exist: {input_shp}')

    outdir = args.outdir
    if outdir is None:
        outdir = input_shp.parent / f'{input_shp.stem}_split_rasterized'

    raster_tifs = rasterize_polygon_tiles(
        input_shp, outdir=outdir, tile_size=args.tile_size,
        overlap_ratio=args.overlap_ratio, pixel_size=args.pixel_size,
        burn_value=args.burn_value)

    dummy_tif = None
    if not args.no_dummy:
        dummy_tif = create_dummy_tif(
            outdir / 'global_dummy.tif', bounds=(-180, -90, 180, 90),
            pixel_size=args.dummy_pixel_size, value=0, dtype='uint8')

    print(f'Generated {len(raster_tifs)} surrogate raster tile(s) in {outdir}')
    for tif_path in raster_tifs:
        print(tif_path)
    if dummy_tif is not None:
        print(f'Background TIF: {dummy_tif}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
