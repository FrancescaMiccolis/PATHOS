#%%
#Env creation
#If you do not have an environment, you can just use the txt file of requirements with
#python3.10 --version --> if you do not have python 3.10, install it to continue with the next steps
#python 3.10 -m venv name_of_the_env 
#source name_of_the_env/bin/activate
#pip install -r requirements.txt

#If you already have a working environment you can just install cellseg-gsontools==0.1.6 
#and the packages imported that are not yet in your environment

#Once you have your environment: change the sys.append() line according to you env path 

#%%
import warnings
from functools import partial
from pathlib import Path
import glob
import os
from typing import List, Tuple
import geopandas as gpd
import palettable.matplotlib as palmpl
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import pandas as pd
from libpysal.cg import alpha_shape_auto
import sys
import argparse
# from legendgram import legendgram
from palettable.cartocolors import qualitative
sys.path.append("path_to_env_with_cellseg_gsontools/lib/python3.10/site-packages/cellseg_gsontools")

from cellseg_gsontools.apply import gdf_apply
from cellseg_gsontools.diversity import local_diversity
from cellseg_gsontools.graphs import fit_graph
from cellseg_gsontools.neighbors import neighborhood, nhood_type_count, nhood_vals
from cellseg_gsontools.utils import set_uid, read_gdf
from cellseg_gsontools.character import local_distances
from cellseg_gsontools.clustering import cluster_points,cluster_cells
from cellseg_gsontools.geometry import shape_metric
from cellseg_gsontools.spatial_context.ops import get_interface_zones
import numpy as np

warnings.filterwarnings("ignore")
#%%
# tissues = read_gdf(sample_meta["area_files"].iloc[0])
# cells = read_gdf(sample_meta["cell_files"].iloc[0])
# cells.set_crs(4328, inplace=True, allow_override=True)
# tissues.set_crs(4328, inplace=True, allow_override=True)

def filtering_att_levels(tissues,cells, attentionmap,level,name,results_path,model):
    att_gdf =read_gdf(attentionmap)
    cell_gdf =read_gdf(
        fn_cell_gdf)
    cell_gdf = set_uid(cell_gdf)
    area_gdf =read_gdf(
        fn_area_gdf)
    print(area_gdf.columns,cell_gdf.columns)
    print(area_gdf.crs,cell_gdf.crs)
    area_gdf.set_crs(cell_gdf.crs, inplace=True, allow_override=True)
    att_gdf.set_crs(cell_gdf.crs, inplace=True, allow_override=True)
    att_x = min(att_gdf["geometry"].bounds["minx"])
    att_y = min(att_gdf["geometry"].bounds["miny"])
    he_x = min(area_gdf["geometry"].bounds["minx"])
    he_y = min(area_gdf["geometry"].bounds["miny"])
    he_x_cell = min(cell_gdf["geometry"].bounds["minx"])
    he_y_cell = min(cell_gdf["geometry"].bounds["miny"])
    area_gdf["geometry"] = area_gdf.translate((-he_x+att_x), (-he_y+att_y))
    print(area_gdf.shape)
    cell_gdf["geometry"] = cell_gdf.translate((-he_x_cell+att_x), (-he_y_cell+att_y))
    cmap = plt.get_cmap('viridis')
    
    high_att_gdf = att_gdf[att_gdf["class_name"] < level]
    print(name)
    print("attentions:", att_gdf.shape, high_att_gdf.shape)
    
    fig,ax = plt.subplots(3,3,figsize=(20,20))
    area_gdf.plot(ax=ax[0,0], column="class_name",aspect=1,legend=True,legend_kwds={
            "fontsize": 8,
            "loc": "center left",
            "bbox_to_anchor": (1.0, 0.94),
        },)
    ax[0,0].set_title('Areas')
    # # legendgram(ax[0,0],area_gdf, "class_name",loc="lower left"),bbox_to_anchor=(1.0, 0.94))
    
    
    cell_gdf.plot(ax=ax[0,1], column="class_name",aspect=1,legend=True,legend_kwds={
            "fontsize": 8,
            "loc": "center left",
            "bbox_to_anchor": (1.0, 0.94),
        },cmap=cmap)
    ax[0,1].set_title('Cells')
    # # ax[0,1]=legendgram(ax[0,1],cell_gdf, "class_name",loc="lower left",bbox_to_anchor=(1.0, 0.94))
   
   
    att_gdf.plot(ax=ax[0,2],column="class_name",aspect=1,legend=True,
                 #legend_kwds={# "fontsize": 8,    "loc": "center left",
    #         "bbox_to_anchor": (1.0, 0.94),
    #    }
    )
    ax[0,2].set_title('Attentions')
    nbins=5
    breaks = np.linspace(att_gdf["class_name"].min(), att_gdf["class_name"].max(), nbins+1 )
    # #legendgram(ax[0,2],att_gdf, "class_name",loc="lower left",bins=nbins,legend_size=(.5,.2), breaks=breaks,pal=palmpl.Viridis_6)
    
    if level < 6:
        att_cell = cell_gdf.clip(high_att_gdf.dissolve())
        att_area = area_gdf.clip(high_att_gdf.dissolve()).explode()
        # Perform spatial joins and reset indices with new names
        att_area_intersect = gpd.sjoin(area_gdf, high_att_gdf, op='intersects').rename(columns={'class_name_left':'class_name'})
        att_cell_intersect = gpd.sjoin(cell_gdf, high_att_gdf, op='intersects').rename(columns={'class_name_left':'class_name'})
        print("intersection areas:", att_area_intersect.columns)
        print(att_area_intersect.shape)
        print("intersection cells:", att_cell_intersect.columns)
        print(att_cell_intersect.shape)
    else:
        att_cell=cell_gdf
        att_area=area_gdf
        att_area_intersect=area_gdf
        att_cell_intersect=cell_gdf
    print("cells:", cell_gdf.shape, att_cell.shape)
    print("areas:", area_gdf.shape, att_area.shape)
    print("Intersected cell geometries:", att_cell_intersect.shape)
    print("Intersected area geometries:", att_area_intersect.shape)

    
    att_area.plot(ax=ax[1,0], aspect=1, column="class_name", legend=True,legend_kwds={
            "fontsize": 8,
            "loc": "center left",
            "bbox_to_anchor": (1.0, 0.94),
        },)
    ax[1,0].set_title('High attentioned areas')
    # # ax[1,0]=legendgram(ax[1,0],att_gdf, "class_name",loc="lower left",bbox_to_anchor=(1.0, 0.94))

    att_cell.plot(ax=ax[1,1], column="class_name",aspect=1,legend=True,legend_kwds={
            "fontsize": 8,
            "loc": "center left",
            "bbox_to_anchor": (1.0, 0.94),
        },cmap=cmap)
    ax[1,1].set_title('High attentioned cells')
    # # ax[1,1]=legendgram(ax[1,1],att_cell, "class_name",loc="lower left",bbox_to_anchor=(1.0, 0.94))
    
    high_att_gdf.plot(ax=ax[1,2],column="class_name",aspect=1,legend=True,
                      #legend_kwds={
            #"fontsize": 8,
        #     "loc": "center left",
        #     "bbox_to_anchor": (1.0, 0.94),
        # },
    )
    
    nbins=4
    breaks = np.linspace(att_gdf["class_name"].min(), att_gdf["class_name"].max(), nbins+1 )
    ax[1,2].set_title('High attentioned attentions')
    # legendgram(ax[1,2],high_att_gdf, "class_name",loc="lower left",bins=nbins,legend_size=(.5,.2), breaks=breaks)

    
    att_area_intersect.plot(ax=ax[2,0], aspect=1, column="class_name", legend=True,legend_kwds={
            "fontsize": 8,
            "loc": "center left",
            "bbox_to_anchor": (1.0, 0.94),
        },)
    ax[2,0].set_title('Intersected attentioned areas')
    # #ax[2,0]=legendgram(ax[2,0],att_area_intersect, "class_name",loc="lower left",bbox_to_anchor=(1.0, 0.94))

    att_cell_intersect.plot(ax=ax[2,1], column="class_name",aspect=1,legend=True,legend_kwds={
            "fontsize": 8,
            "loc": "center left",
            "bbox_to_anchor": (1.0, 0.94),
        },cmap=cmap)
    ax[2,1].set_title('Intersected attentioned cells')
    # #ax[2,1]=legendgram(ax[2,1],att_cell_intersect, "class_name",loc="lower left",bbox_to_anchor=(1.0, 0.94))
    

    high_att_gdf.plot(ax=ax[2,2],column="class_name",aspect=1,legend=True)
        #               ,legend_kwds={
        #     "fontsize": 8,
        #     "loc": "center left",
        #     "bbox_to_anchor": (1.0, 0.94),
        # },)
    ax[2,2].set_title('Intersected attentioned attentions')
    nbins=4
    breaks = np.linspace(att_gdf["class_name"].min(), att_gdf["class_name"].max(), nbins+1 )
    # legendgram(ax[2,2],high_att_gdf, "class_name",bins=nbinds,legend_size=(.5,.2), breaks=breaks)
    #ax.set_aspect('equal')
    
    # plt.tight_layout()
    # if not(os.path.exists(os.path.join(results_path,'figures',model))):
    #     os.makedirs(os.path.join(results_path,'figures',model))
    # plt.savefig(os.path.join(results_path,'figures',model,f'{name}.png'), bbox_inches='tight', dpi=300)
    return att_area,att_cell,att_area_intersect,att_cell_intersect

#%%
def get_index(cells_in_region, w, cell_type):
    # Get the neihgboring nodes of the graph
    func = partial(neighborhood, spatial_weights=w)
    if cells_in_region.shape[0] !=0:
        cells_in_region["nhood"] = gdf_apply(cells_in_region, func, columns=["uid"])

        # Get the classes of the neighboring nodes
        func = partial(nhood_vals, values=cells_in_region["class_name"])
        cells_in_region["nhood_classes"] = gdf_apply(
            cells_in_region,
            func=func,
            parallel=True,
            columns=["nhood"],
        )

        # Get the number of inflammatory cells in the neighborhood
        func = partial(nhood_type_count, cls=cell_type, frac=False)
        cells_in_region[f"{cell_type}_cnt"] = gdf_apply(
            cells_in_region,
            func=func,
            parallel=True,
            columns=["nhood_classes"],
        )

        # Get the fraction of inflammatory cells in the neighborhood
        func = partial(nhood_type_count, cls=cell_type, frac=True)
        cells_in_region[f"{cell_type}_frac"] = gdf_apply(
            cells_in_region,
            func=func,
            parallel=True,
            columns=["nhood_classes"],
        )

        # This will smooth the extremes (e.g. if there is only one inflammatory cell in the
        # neighborhood, the fraction will be 1)

        cells_in_region[f"{cell_type}_index"] = (
            cells_in_region[f"{cell_type}_frac"] * cells_in_region[f"{cell_type}_cnt"]
        )

        return cells_in_region
    else:
        return pd.Series()

#neoplastic as celltype with inflammatory,connective,macrophages
#%%
def interaction_pipe(tissues, cells, tissue: str, cell_type: str, focal_type: str):
    ############
    celltype_nhood_feats = {
        f"mean_nhood_{cell_type}_index_in_{tissue}": 0,
        f"mean_nhood_{cell_type}_frac_in_{tissue}": 0,
        f"mean_nhood_{cell_type}_cnt_in_{tissue}": 0,
        f"mean_nhood_class_simpson_in_{tissue}": 0,
        f"mean_nhood_{focal_type}_{cell_type}_cnt_in_{tissue}": 0,
        f"mean_nhood_{focal_type}_{cell_type}_index_in_{tissue}": 0,
        f"mean_nhood_{focal_type}_{cell_type}_frac_in_{tissue}": 0,
    }
    region = tissues[tissues["class_name"] == tissue]
    region.set_crs(4328, inplace=True, allow_override=True)

    _, cell_inds = cells.sindex.query(region.geometry, predicate="intersects")
    cells_in_region = cells.iloc[np.unique(cell_inds)]
    cells_in_region = cells_in_region[["geometry", "class_name"]]
    cells_in_region.reset_index(drop=True, inplace=True)
    cells_in_region.set_crs(4328, inplace=True, allow_override=True)

    cells_in_region = set_uid(cells_in_region, id_col="uid", drop=False)

    # Fit the distband
    w = fit_graph(
        cells_in_region,
        type="distband",
        id_col="uid",
        thresh=128,
    )
    if w is not None:
        # Row-standardized weights
        w.transform = "R"

        # Get the neihgboring nodes of the graph
        cells_in_region = get_index(cells_in_region, w, cell_type)

        # This will smooth the extremes (e.g. if there is only one inflammatory cell in the
        # neighborhood, the fraction will be 1)
        cells_in_region[f"{cell_type}_index"] = (
            cells_in_region[f"{cell_type}_frac"] * cells_in_region[f"{cell_type}_cnt"]
        )

        cells_in_region = local_diversity(
            cells_in_region,
            spatial_weights=w,
            val_col="class_name",
            metrics=["simpson_index"],
        )

        celltype_nhood_feats[f"mean_nhood_{cell_type}_index_in_{tissue}"] = cells_in_region[
            f"{cell_type}_index"
        ].mean()
        celltype_nhood_feats[f"mean_nhood_{cell_type}_frac_in_{tissue}"] = cells_in_region[
            f"{cell_type}_frac"
        ].mean()

        celltype_nhood_feats[f"mean_nhood_{cell_type}_cnt_in_{tissue}"] = cells_in_region[
            f"{cell_type}_cnt"
        ].mean()

        celltype_nhood_feats[f"mean_nhood_class_simpson_in_{tissue}"] = cells_in_region[
            "class_name_simpson_index"
        ].mean()

        dd = cells_in_region[cells_in_region["class_name"].isin([cell_type, focal_type])]
        dd = set_uid(dd, id_col="uid", drop=False)

        # Fit the distband
        w = fit_graph(
            dd,
            type="distband",
            id_col="uid",
            thresh=128,
        )

        dd = get_index(dd, w, cell_type)
        if dd.shape[0] != 0:
            celltype_nhood_feats[f"mean_nhood_{focal_type}_{cell_type}_index_in_{tissue}"] = dd[
                f"{cell_type}_index"
            ].mean()
            celltype_nhood_feats[f"mean_nhood_{focal_type}_{cell_type}_frac_in_{tissue}"] = dd[
                f"{cell_type}_frac"
            ].mean()

            celltype_nhood_feats[f"mean_nhood_{focal_type}_{cell_type}_cnt_in_{tissue}"] = dd[
                f"{cell_type}_cnt"
            ].mean()

        # Row-standardized weights
            w.transform = "R"
            dd = local_diversity(
                dd,
                spatial_weights=w,
                val_col="class_name",
                metrics=["simpson_index"],
            )
            celltype_nhood_feats[
                f"mean_nhood_{focal_type}_{cell_type}_simpson_in_{tissue}"
            ] = dd["class_name_simpson_index"].mean()

            return pd.Series(celltype_nhood_feats)
        else:
            return pd.Series()
    else:
        return pd.Series()

TO_MM_CONVERSION = 0.5 / 1e3  # conversion to mm
TO_MM_SQUARED_CONVERSION = 0.125 / 1e6  # conversion to mm^2


def get_clust_stat_ret_dict():
    return {
        "adjascent_tumor_immune_clust_dispersion_mean": 0.0,
        "adjascent_tumor_immune_clust_dispersion_std": 0.0,
        "adjascent_tumor_immune_clust_dispersion_max": 0.0,
        "distal_immune_clust_dispersion_mean": 0.0,
        "distal_immune_clust_dispersion_std": 0.0,
        "distal_immune_clust_dispersion_max": 0.0,
        "adjascent_tumor_immune_clust_size_mean": 0.0,
        "adjascent_tumor_immune_clust_size_std": 0.0,
        "adjascent_tumor_immune_clust_size_max": 0.0,
        "distal_immune_clust_size_mean": 0.0,
        "distal_immune_clust_size_std": 0.0,
        "distal_immune_clust_size_max": 0.0,
        "adjascent_tumor_immune_clust_area_mean": 0.0,
        "adjascent_tumor_immune_clust_area_std": 0.0,
        "adjascent_tumor_immune_clust_area_max": 0.0,
        "distal_immune_clust_area_mean": 0.0,
        "distal_immune_clust_area_std": 0.0,
        "distal_immune_clust_area_max": 0.0,
    }


def get_objs(objs: gpd.GeoDataFrame, area: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Get the objects that are in the given area.

    Uses the spatial index to find the objects that are in the area (high performance).

    Parameters:
        objs (gpd.GeoDataFrame):
            The GeoDataFrame with the objects.
        area (gpd.GeoDataFrame):
            The GeoDataFrame with the area.

    Returns:
        objs_in_area (gpd.GeoDataFrame):
            The objects in the area.
    """
    _, obj_inds = objs.sindex.query(area.geometry, predicate="intersects")
    objs_in_area = objs.iloc[np.unique(obj_inds)]

    return objs_in_area

def get_cluster_dispersion(cells: gpd.GeoDataFrame) -> float:
    """Calculate the dispersion of the given cells.

    The dispersion is calculated as the square root of the average of the squared
    distances from the centroid.

    Parameters:
        cells (gpd.GeoDataFrame):
            The GeoDataFrame with the cells.

    Returns:
        dispersion (float):
            The dispersion of the cells.
    """
    xy = np.vstack([cells.centroid.x, cells.centroid.y]).T
    n, p = xy.shape
    m = xy.mean(axis=0)
    return np.sqrt(((xy * xy).sum(axis=0) / n - m * m).sum())


def get_cluster_area(cells: gpd.GeoDataFrame) -> float:
    """Calculate the area of the given cells.

    The area is calculated as the area of the alpha shape of the cells.

    Parameters:
        cells (gpd.GeoDataFrame):
            The GeoDataFrame with the cells.

    Returns:
        area (float):
            The area of the cells.
    """
    coords = np.vstack([cells.centroid.x, cells.centroid.y]).T
    alpha_shape = alpha_shape_auto(coords, step=15)
    return alpha_shape.area

def tissue_interfacet(
    area_gdf: gpd.GeoDataFrame,
    cell_gdf: gpd.GeoDataFrame,
    tissue_type: str = "area_cin",
    stroma_types: List[str] = ["areastroma", "blood"],
    buf_dist: int = 500,
) -> gpd.GeoDataFrame:
    """Get the interface areas between tissues.

    Parameters:
        area_gdf (gpd.GeoDataFrame):
            The GeoDataFrame with the areas.
        cell_gdf (gpd.GeoDataFrame):
            The GeoDataFrame with the cells.
        tissue_type (str):
            The class name of the tissue.
        stroma_types (List[str]):
            The class names of the stroma.
        buf_dist (int):
            The buffer distance for the interface zone.
    Returns:
        iface (gpd.GeoDataFrame):
            The GeoDataFrame with the interface zones.
    """
    cells = cell_gdf.copy()

    # Get the tissue and stroma
    tissue = area_gdf.loc[area_gdf["class_name"] == tissue_type]
    tissue.set_crs(4328, inplace=True, allow_override=True)
    tissue = tissue.loc[tissue.area > 1e5]

    stroma = area_gdf.loc[area_gdf["class_name"].isin(stroma_types)]
    stroma.set_crs(4328, inplace=True, allow_override=True)

    # Get the interface zones
    iface = get_interface_zones(tissue, stroma, buffer_dist=buf_dist)

    if iface.empty:
        return cells, iface

    iface = iface.dissolve().explode(index_parts=False)
    return iface

def cluster_stroma_context(
    area_gdf: gpd.GeoDataFrame,
    cluster_gdf: gpd.GeoDataFrame,
    tissue_type: str = "area_cin",
    stroma_types: List[str] = ["areastroma", "blood"],
    buf_dist: int = 500,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Get the stromal context of the given clustered cells.

    The stromal context is defined as:
    - 'distal': The cluster is not in the interface zones.
    - 'adjascent': The cluster is in the immediate interface zone.
    - 'proximal': The cluster is in the further away interface zone.

    Parameters:
        area_gdf (gpd.GeoDataFrame):
            The GeoDataFrame with the areas.
        cluster_gdf (gpd.GeoDataFrame):
            The GeoDataFrame with the clustered cells.
        tissue_type (str):
            The class name of the tissue.
        stroma_types (List[str]):
            The class names of the stroma.
        buf_dist (int):
            The buffer distance for the interface zone.

    Returns:
        clustered_cells (gpd.GeoDataFrame):
            The GeoDataFrame with the clustered cells.
        iface (gpd.GeoDataFrame):
            The GeoDataFrame with the interface zones.
    """
    clustered_cells = cluster_gdf.copy()
    clustered_cells = clustered_cells.assign(stromal_context="distal")

    # Get the tissue and stroma
    tissue = area_gdf.loc[area_gdf["class_name"] == tissue_type]
    tissue.set_crs(4328, inplace=True, allow_override=True)
    tissue = tissue.loc[tissue.area > 1e5]

    stroma = area_gdf.loc[area_gdf["class_name"].isin(stroma_types)]
    stroma.set_crs(4328, inplace=True, allow_override=True)

    # Get the interface zones
    iface = get_interface_zones(tissue, stroma, buffer_dist=buf_dist)

    if iface.empty:
        return clustered_cells, iface

    iface = iface.dissolve().explode(index_parts=False)

    # assign the stromal context
    for lab in sorted(clustered_cells.label.unique()):
        clust = clustered_cells.loc[clustered_cells["label"] == lab]

        # Get the cells in the interface zones
        clust_iface = get_objs(clust, iface)

        # Assign the stromal context, if over 15% cluster extends to the interface zones
        # the stromal context is set to adjascent. Adjascent is the immediate interface
        # zone, If the cluster is not in the interface zones, the stromal context is set
        # to distal.
        if len(clust_iface) / len(clust) > 0.15:
            clust = clust.assign(stromal_context="adjascent")

        clustered_cells.loc[clust.index, "stromal_context"] = clust["stromal_context"]

    return clustered_cells, iface


def context4clusters(
    clustered_cells: gpd.GeoDataFrame, area_gdf: gpd.GeoDataFrame
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Find the stromal contexts of the cell clusters.

    The stromal contexts are defined as:
    - 'distal': The cluster is not in the interface zones.
    - 'adjascent': The cluster is in the immediate interface zone.
    - 'proximal': The cluster is in the further away interface zone.

    Parameters:
        cell_gdf (gpd.GeoDataFrame):
            The GeoDataFrame with the cells.
        area_gdf (gpd.GeoDataFrame):
            The GeoDataFrame with the areas.

    Returns:
        final_clusts (gpd.GeoDataFrame):
            The GeoDataFrame with the clustered cells and the stromal contexts.
        tumor_iface (gpd.GeoDataFrame):
            The GeoDataFrame with the interface zones of the tumor areas.
        squam_iface (gpd.GeoDataFrame):
            The GeoDataFrame with the interface zones of the squamous areas.
        gland_iface (gpd.GeoDataFrame):
            The GeoDataFrame with the interface zones of the gland areas.
    """
    tumor_clusts, tumor_iface = cluster_stroma_context(
        area_gdf=area_gdf,
        cluster_gdf=clustered_cells,
        tissue_type="area_cin",
        stroma_types=["areastroma", "blood"],
        buf_dist=1000,
    )
    # Refine the stromal contexts based on the interface zones

    # Start with a copy of tumor_clusts
    final_clusts = tumor_clusts.copy()
    final_clusts.loc[
        final_clusts["stromal_context"] == "adjascent", "stromal_context"
    ] = "adjascent_tumor"

    # Ensure 'distal' values that are set as 'distal' in all of the three gdfs are
    # left as 'distal'
    mask = tumor_clusts["stromal_context"] == "distal"
    final_clusts.loc[mask, "stromal_context"] = "distal"

    return final_clusts, tumor_iface


def plot_intersect(
    area_gdf: gpd.GeoDataFrame,
    area: gpd.GeoDataFrame,
    intersect: gpd.GeoDataFrame,
    name: str,
    fig_save_dir: Path,
) -> None:
    ax = (
        area.dissolve()
        .explode(index_parts=False)
        .plot(color="red", figsize=(10, 10), alpha=0.5, aspect=1)
    )
    ax = area_gdf.plot(ax=ax, column="class_name", alpha=0.2, aspect=1)
    ax = intersect.plot(ax=ax, color="green", alpha=0.8, aspect=1)
    ax.set_axis_off()
    ax.set_title(f"{name} Intersection", fontsize=25)
    ax.figure.savefig(
        fig_save_dir / f"{name}_tissue_intersection.png",
    )
    plt.cla()
    plt.close("all")
    ax.figure.clf()


def plot_immune_clusters(
    area_gdf: gpd.GeoDataFrame,
    tumor_iface: gpd.GeoDataFrame,
    final_clusts: gpd.GeoDataFrame,
    name: str,
    fig_save_dir: Path,
) -> None:
    ax = area_gdf.plot(aspect=1, column="class_name", alpha=0.2, figsize=(20, 20))
    ax = tumor_iface.plot(ax=ax, aspect=1, color="red", alpha=0.1)
    ax = final_clusts.plot(
        ax=ax,
        column="stromal_context",
        legend=True,
        categorical=True,
        aspect=1,
        cmap="brg",
    )
    ax.set_axis_off()
    ax.set_title(f"{name} Immune Clusters", fontsize=25)
    ax.figure.savefig(
        fig_save_dir / f"{name}_immune_clusters.png",
    )
    plt.cla()
    plt.close("all")
    ax.figure.clf()
#%%

##################################################################
##################################################################
########################### PIPELINE #############################
##################################################################
# cluster immune cells and get the stromal context of the clusters
def feature_extraction_pipeline(
    cell_gdf: gpd.GeoDataFrame,
    area_gdf: gpd.GeoDataFrame,
    do_plots: bool = False,
    name: str = "",
    fig_save_dir: Path = Path("."),
) -> pd.Series:
    area_gdf = area_gdf.loc[~area_gdf["class_name"].isna()]
    cell_gdf = cell_gdf.loc[~cell_gdf["class_name"].isna()]
    cell_gdf = set_uid(cell_gdf, id_col="uid", drop=False)
    try:
        clustered_cells = cluster_cells(cell_gdf, cell_type="inflammatory", seed=123)
    except:
        clustered_cells=pd.Series()
    if clustered_cells.shape[0] != 0:
        try:  
            final_clusts, tumor_iface = context4clusters(clustered_cells, area_gdf)
            TSI_TILS = final_clusts.loc[final_clusts["stromal_context"] == "adjascent_tumor"]
            distal_immune_cluster_cells = final_clusts.loc[
                final_clusts["stromal_context"] == "distal"
            ]
        except:
            final_clusts=pd.Series()
            tumor_iface=pd.Series()
            TSI_TILS=pd.Series()
            distal_immune_cluster_cells = pd.Series()
            # Get the different tissue areas
        stroma = area_gdf.loc[area_gdf["class_name"].isin(["areastroma", "areahemorragia"])]
        stroma.set_crs(4328, inplace=True, allow_override=True)

        tumor = area_gdf.loc[area_gdf["class_name"] == "area_cin"]
        tumor.set_crs(4328, inplace=True, allow_override=True)

        # Get the different cells from the different tissue areas
        tumor_cells = cell_gdf.loc[cell_gdf["class_name"] == "neoplastic"]
        immune_cells = cell_gdf.loc[cell_gdf["class_name"] == "inflammatory"]
        stromal_cells = cell_gdf.loc[cell_gdf["class_name"] == "connective"]

        # Get immune cells in different tissue contexts
        tumor_cells: gpd.GeoDataFrame = get_objs(tumor_cells, tumor)
        TIL_cells: gpd.GeoDataFrame = get_objs(immune_cells, tumor)
        stromal_immune_cells: gpd.GeoDataFrame = get_objs(immune_cells, stroma)

        # compute shape metrics for the tumor cells
        if tumor_cells.empty:
            tumor_cells=pd.Series()
        else:
            tumor_cells = shape_metric(
            tumor_cells,
            metrics=[
                "area",
                "major_axis_len",
                "eccentricity",
                "fractal_dimension",
                ],
            )

            tumor_cells = set_uid(tumor_cells, id_col="uid", drop=False)
            w = fit_graph(tumor_cells, "distband", "uid", thresh=100)
            tumor_cells = local_distances(tumor_cells, w, id_col="uid", reductions=["mean"])

        ##################################################################
        ##################################################################
        ####################### STATS AGGREGATION ########################
        ##################################################################

        if do_plots:
            plot_immune_clusters(
                area_gdf, tumor_iface, final_clusts, name, fig_save_dir / "immune_clusters"
            )
        else:
            
            if final_clusts.empty or stromal_immune_cells.empty:
                clust_immune_propotion = 0
            else:
                # Get the proportion of immune cells in the different stromal contexts
                clust_immune_propotion = len(final_clusts) / len(stromal_immune_cells)
            if TSI_TILS.empty or stromal_immune_cells.empty:
                tumor_clust_immune_proportion = 0
            else:
                tumor_clust_immune_proportion = len(TSI_TILS) / len(stromal_immune_cells)

            # Get the cell type proportions
            if immune_cells.empty or cell_gdf.empty:
                immune_cell_propotion = 0
            else:
                immune_cell_propotion = len(immune_cells) / len(cell_gdf)
            if stromal_cells.empty or cell_gdf.empty:
                stromal_cell_propotion = 0
            else:   
                stromal_cell_propotion = len(stromal_cells) / len(cell_gdf)
            if tumor_cells.empty or cell_gdf.empty:
                tumor_cell_propotion = 0
                tumor_cell_shape_props = {}
                tumor_cell_density_props = {}
            else:
                tumor_cell_propotion = len(tumor_cells) / len(cell_gdf)

                # Compute tumor cell shape metrics
                tumor_cell_shape_props = tumor_cells[
                    ["area", "major_axis_len", "eccentricity", "fractal_dimension"]
                ]
                tumor_cell_shape_props["area"] *= TO_MM_SQUARED_CONVERSION
                tumor_cell_shape_props["major_axis_len"] *= TO_MM_CONVERSION
                tumor_cell_shape_props = tumor_cell_shape_props.apply(
                    lambda x: pd.Series(
                        {"mean": np.mean(x), "std": np.std(x), "max": np.max(x)}
                    )
                ).add_prefix("tumor_cells_")
                tumor_cell_shape_props = tumor_cell_shape_props = {
                    "_".join([met, red]): val
                    for met, vals in tumor_cell_shape_props.to_dict().items()
                    for red, val in vals.items()
                }

                # Compute tumor cell distance metrics
                tumor_cell_density_props = tumor_cells[["nhood_dists_mean"]] * TO_MM_CONVERSION
                tumor_cell_density_props = tumor_cell_density_props.apply(
                    lambda x: pd.Series({"mean": np.mean(x), "std": np.std(x)})
                ).add_prefix("tumor_cells_")
                tumor_cell_density_props = tumor_cell_density_props = {
                    "_".join([met, red]): val
                    for met, vals in tumor_cell_density_props.to_dict().items()
                    for red, val in vals.items()
                }

            # Get the area of the different stromal contexts and tissues
            stroma_area = stroma.area.sum()
            tumor_area = tumor.area.sum()
            if tumor_iface.empty:
                TSI_area = 0
                distal_stroma_area = stroma_area
            else:
                TSI_area = tumor_iface.area.sum()
                distal_stroma_area = stroma_area - TSI_area

            # Get the immune densities
            if TIL_cells.empty:
                TIL_density = 0
            else:
                TIL_density = (
                len(TIL_cells) / (tumor.area.sum() * TO_MM_SQUARED_CONVERSION)
                if tumor_area > 0
                else 0
                )
            if stroma_area == 0 or stromal_immune_cells.empty :
                stromal_immune_density = 0
            else:
                stromal_immune_density = (
                    len(stromal_immune_cells) / (stroma_area * TO_MM_SQUARED_CONVERSION)
                    if stroma_area > 0
                    else 0
                )
            if TSI_TILS.empty or stroma_area == 0:
                
                TSI_immune_density_to_stroma = 0
            else:
            # Get the immune density to stromal area
                TSI_immune_density_to_stroma = (
                    len(TSI_TILS) / (stroma_area * TO_MM_SQUARED_CONVERSION)
                    if stroma_area > 0
                    else 0
                )
            if TSI_TILS.empty or TSI_area == 0 :
                TSI_immune_density = 0
            else:
            # Get the immune density for each stromal context
                TSI_immune_density = (
                    len(TSI_TILS) / (TSI_area * TO_MM_SQUARED_CONVERSION) if TSI_area > 0 else 0
                    )
            if distal_immune_cluster_cells.empty or distal_stroma_area == 0:
                distal_immune_density = 0
            else:
                
                distal_immune_density = (
                    len(final_clusts.loc[final_clusts["stromal_context"] == "distal"])
                    / (distal_stroma_area * TO_MM_SQUARED_CONVERSION)
                    if distal_stroma_area > 0
                    else 0
                )

            # Compute the cluster dispersion, area and size for each cluster in each stromal context
            if final_clusts.empty:
                cluster_dispersion = pd.Series()
                cluster_area = pd.Series()
                cluster_size = pd.Series()
                cluster_stats = get_clust_stat_ret_dict()
            else:
                cluster_dispersion = (
                    final_clusts.groupby(["stromal_context", "label"])
                    .apply(
                        lambda x: get_cluster_dispersion(x["geometry"]), include_groups=False
                    )
                    .add_suffix("_immune_clust_dispersion")
                )
                cluster_area = (
                    final_clusts.groupby(["stromal_context", "label"])
                    .apply(lambda x: get_cluster_area(x["geometry"]), include_groups=False)
                    .add_suffix("_immune_clust_area")
                ) * TO_MM_SQUARED_CONVERSION
                cluster_size = (
                    final_clusts.groupby(["stromal_context", "label"])
                    .apply(lambda x: len(x), include_groups=False)
                    .add_suffix("_immune_clust_size")
                )

                # get stats for each cluster at different stromal contexts
                cluster_stats = get_clust_stat_ret_dict()
                cluster_dispersion_stats = cluster_dispersion.groupby("stromal_context").apply(
                    lambda x: pd.Series(
                        {"mean": np.mean(x), "std": np.std(x), "max": np.max(x)}
                    )
                )
                cluster_dispersion_stats = {
                    "_".join(ind): val for ind, val in cluster_dispersion_stats.items()
                }
                for k, i in cluster_dispersion_stats.items():
                    cluster_stats[k] = i

                cluster_size_stats = cluster_size.groupby("stromal_context").apply(
                    lambda x: pd.Series(
                        {"mean": np.mean(x), "std": np.std(x), "max": np.max(x)}
                    )
                )
                cluster_size_stats = {
                    "_".join(ind): val for ind, val in cluster_size_stats.items()
                }
                for k, i in cluster_size_stats.items():
                    cluster_stats[k] = i

                cluster_area_stats = cluster_area.groupby("stromal_context").apply(
                    lambda x: pd.Series(
                        {"mean": np.mean(x), "std": np.std(x), "max": np.max(x)}
                    )
                )
                cluster_area_stats = {
                    "_".join(ind): val for ind, val in cluster_area_stats.items()
                }
                for k, i in cluster_area_stats.items():
                    cluster_stats[k] = i
            if TSI_TILS.empty:
                tumor_n_clusters = 0
            else:
            # Get number of cluster for each stromal context
                tumor_n_clusters = len(TSI_TILS["label"].unique())

            ret = {
                "tumor_n_cells": len(tumor_cells),
                "immune_n_cells": len(immune_cells),
                "stromal_n_cells": len(stromal_cells),
                "TSI_n_immune_cells": len(TSI_TILS),
                "distal_n_immune_cells": len(distal_immune_cluster_cells),
                "TIL_n_cells": len(TIL_cells),
                "TSI_clustered_immune_density": TSI_immune_density,
                "distal_stroma_clustered_immune_density": distal_immune_density,
                "TSI_immune_density_to_stroma": TSI_immune_density_to_stroma,
                "TIL_density": TIL_density,
                "stromal_immune_density": stromal_immune_density,
                "clustered_immune_cell_proportion": clust_immune_propotion,
                "tumor_clust_immune_proportion": tumor_clust_immune_proportion,
                "stroma_area": stroma_area * TO_MM_SQUARED_CONVERSION,
                "tumor_area": tumor_area * TO_MM_SQUARED_CONVERSION,
                "TSI_area": TSI_area * TO_MM_SQUARED_CONVERSION,
                "distal_stroma_area": distal_stroma_area * TO_MM_SQUARED_CONVERSION,
                "tumor_cell_proportion": tumor_cell_propotion,
                "immune_cell_proportion": immune_cell_propotion,
                "stromal_cell_proportion": stromal_cell_propotion,
                "TSI_n_immune_clusters": tumor_n_clusters,
            }

            return pd.Series(
                {
                    **ret,
                    **tumor_cell_shape_props,
                    **tumor_cell_density_props,
                    **cluster_stats,
                }
            )
    else: 
        return pd.Series()

#%%
def convert_to_polygon(geom):
    delta=10
    if geom.type in ['Polygon', 'GeometryCollection']:
        return geom
    elif geom.type == 'Point':
        # Convert Point or MultiPoint to a Polygon with a single point
        return geom.buffer(delta)
    elif geom.type == 'LineString':
        # Convert LineString or MultiLineString to a Polygon with the coordinates of the LineString
        coords = list(geom.coords)
        if len(coords) >= 3:
            if coords[0] != coords[-1]:
                coords.append(coords[0])
            return Polygon(coords)
        elif len(coords) == 2:
            x1, y1 = geom.coords[0]
            x2, y2 = geom.coords[1]
        # Create a small rectangle around the LineString
            return Polygon([(x1 - delta, y1 - delta), (x1 + delta, y1 + delta), 
                        (x2 + delta, y2 + delta), (x2 - delta, y2 - delta), 
                        (x1 - delta, y1 - delta)])       
        else:
            return geom.buffer(delta)
    elif geom.type == 'MultiPolygon':
        # Combine all polygons into a single polygon
        combined_polygon = unary_union(geom)
        return combined_polygon
    elif geom.type == 'MultiPoint':
        return geom.convex_hull 
    elif geom.type == 'MultiLineString':
        # Convert each LineString to a Polygon
        polygons = [convert_to_polygon(line) for line in geom]
        # Combine the resulting polygons into a single polygon
        combined_polygon = unary_union(polygons)
        return combined_polygon
    else:
        pirnt('Geometry type not recognized for',geom)
        return None

#%%
parser=argparse.ArgumentParser()
parser.add_argument("--model",type=str,default="buffermil") #Default model name is buffermil
#Buffermil returns attention maps discretized in 5 levels, 
#where the level 1 contains the most relevant patches and so on to the level 5, that contains the least attentioned ones
#The filtering function take the level as upper threshold to filter out
#Hence, put 6 to obtain the features of the whole slide, without filtering any patches
#If you want to also filter out the least attentioned patches use --level 5, it will keep all the patches with attention between 1 and 4
parser.add_argument("--level",type=int,default=5)   
#Insert the path where to save the extracted features 
parser.add_argument("--results_path",type=str,default="output")
#the data:path is intended as the output path of the panoptic segmentation, 
#with 'cells' and 'areas' folders containing the merged geojson of the panoptic segmentation
parser.add_argument("--data_path",type=str,default="data/input") 
#Intersection is still in test mode, keep it False for now 
parser.add_argument("--intersection",type=bool,default=False)
args=parser.parse_args()
model=args.model
level=args.level
results_path=args.results_path
data_path=args.data_path
intersection=args.intersection


#This is the path containing the json files of the attention maps returned by buffermil, adjust the path according to yours
atts=glob.glob(f"/pathtoattentionmaps/{model}/*/*.json")
print(results_path,level,model)

atts=sorted(atts,key=lambda x:os.path.basename(x))
cell_types = ["neoplastic",
"inflammatory",
"connective",
"dead",
"macrophage_cell",
"macrophage_nucleus"]
tissue_types = ["areastroma",
"areaomentum",
"area_cin",
"areahemorragia",
"necrosis",
"areaserum"]


for element in atts:
    
    name = element.split(".")[0].split("/")[-1]
    name = name.replace("_bestep", "")
    fn_cell_gdf = os.path.join(data_path, 'cells', name+'_cells.geojson')
    fn_area_gdf = os.path.join(data_path, 'areas', name+'_areas.geojson')
    if os.path.exists(os.path.join(results_path,model,name+'.csv')):
        print(model,name,'already done')
        continue
    elif (os.path.exists(fn_area_gdf)) and (os.path.exists(fn_cell_gdf)):
        print(name, "running")
        att_area,att_cell, att_area_intersect, att_cell_intersect = filtering_att_levels(fn_area_gdf,fn_cell_gdf,element,level,name,results_path,model)
        if intersection:
            print('Intersection mode')
            input_area=att_area_intersect.copy()
            input_cell=att_cell_intersect.copy()
        else:
            print('Standard mode')
            input_area=att_area.copy()
            input_cell=att_cell.copy()
        if len(input_area.geometry.geom_type.value_counts()) > 1:
            print(input_area.crs)
            print('Converting geometries to polygons:\n',(input_area.geometry.geom_type.value_counts()))
            input_area['geometry'] = input_area['geometry'].apply(convert_to_polygon)
        if len(input_cell.geometry.geom_type.value_counts()) > 1:
            print(input_cell.crs)
            input_cell['geometry'] = input_cell['geometry'].apply(convert_to_polygon)
        input_cell = input_cell[input_cell['geometry'].notnull()]
        input_area = input_area[input_area['geometry'].notnull()]
        feats = feature_extraction_pipeline(input_cell,input_area)
        neo_conn = interaction_pipe(input_area,input_cell, "area_cin", "neoplastic", "connective")
        neo_macro = interaction_pipe(input_area,input_cell, "area_cin", "neoplastic", "macrophage_cell")
        neo_infl = interaction_pipe(input_area,input_cell, "area_cin", "neoplastic", "inflammatory")
        allfeats=pd.concat([neo_conn,neo_macro,neo_infl,feats])
        print(name,'done. Saving file..')
        if not(os.path.exists(os.path.join(results_path,model))):
            os.makedirs(os.path.join(results_path,model))
        allfeats.to_csv(os.path.join(results_path,model,name+'.csv'))
    else:
        print('No area or cell gdf found for ',model,name)
        continue
# %%
