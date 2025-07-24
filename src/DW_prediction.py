import ee
import datetime
import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
import os
from sklearn.metrics import confusion_matrix
import pandas as pd

DW_CLASSES = [
    "Water", "Trees", "Grass", "Flooded Veg.", "Crops",
    "Shrub & Scrub", "Built Area", "Bare Ground", "Snow & Ice"
]
#*GET THREE DIFFERENT IMAGES WITH 6 BANDS WITH +-2 WEEK(MONTH) , +-3 MONTHS(6MONTHS) , +-6MONTHS(YEAR)*
def run_export_dw_summary_from_consensus(CONSENSUS_SHP, project_name, CONSENSUS_DATE, export_folder_name):
    ee.Initialize(project=project_name)

    consensus_fc = ee.FeatureCollection(CONSENSUS_SHP)
    tile_geom = consensus_fc.geometry()
    date_ee = ee.Date(CONSENSUS_DATE)

    periods = {
        "month": (date_ee.advance(-2, "week"), date_ee.advance(2, "week")),
        "6months": (date_ee.advance(-3, "month"), date_ee.advance(3, "month")),
        "year": (date_ee.advance(-6, "month"), date_ee.advance(6, "month"))
    }

    def max_band_and_index(img):
        arr = img.toArray()
        maxval = arr.arrayReduce(ee.Reducer.max(), [0]).arrayGet([0])
        maxidx = arr.arrayArgmax().arrayGet([0])
        return img.expression('float(idx)', {'idx': maxidx}).rename('max_idx') \
            .addBands(img.expression('float(val)', {'val': maxval}).rename('max_val'))

    def reduce_dw_collection(start, end, geom):
        dw = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1") \
            .filterBounds(geom) \
            .filterDate(start, end) \
            .map(lambda img: img.clip(geom))
        prob_bands = [
            "water", "trees", "grass", "flooded_vegetation", "crops",
            "shrub_and_scrub", "built", "bare", "snow_and_ice"
        ]
        mean_img = dw.select(prob_bands).mean()
        median_img = dw.select(prob_bands).median()
        mean_max = max_band_and_index(mean_img)
        median_max = max_band_and_index(median_img)
        mode_label = dw.select("label").reduce(ee.Reducer.mode()).toFloat()
        count_img = dw.select("label").count().toFloat()
        return mean_max.select('max_idx').rename('mean_max_idx').toFloat() \
            .addBands(mean_max.select('max_val').rename('mean_max_prob').toFloat()) \
            .addBands(median_max.select('max_idx').rename('median_max_idx').toFloat()) \
            .addBands(median_max.select('max_val').rename('median_max_prob').toFloat()) \
            .addBands(mode_label.rename("mode_label")) \
            .addBands(count_img.rename("count"))

    for period, (start, end) in periods.items():
        dw_img = reduce_dw_collection(start, end, tile_geom)
        export_img = dw_img.clip(tile_geom)
        task = ee.batch.Export.image.toDrive(
            image=export_img,
            description=f"DW_{period}_summary",
            folder=export_folder_name,
            fileNamePrefix=f"DW_{period}_summary",
            region=tile_geom,
            scale=10,
            maxPixels=1e13
        )
        task.start()
        print(f"Export started for {period} summary image.")

#*ALLIGN THE DW-IMAGE WITH THE CRS AND META-DATA OF CONSENSUS IMAGE*
def align_dw_prediction_to_consensus(dw_path, consensus_path):
    with rasterio.open(consensus_path) as ref:
        dst_crs = ref.crs
        dst_transform = ref.transform
        dst_width = ref.width
        dst_height = ref.height

    with rasterio.open(dw_path) as src:
        source_data = src.read()
        source_crs = src.crs
        source_transform = src.transform
        source_dtype = src.dtypes[0]
        count = src.count

    destination = np.empty((count, dst_height, dst_width), dtype=source_dtype)

    for b in range(count):
        reproject(
            source=source_data[b],
            destination=destination[b],
            src_transform=source_transform,
            src_crs=source_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_width=dst_width,
            dst_height=dst_height,
            resampling=Resampling.nearest
        )

    file_name = os.path.basename(dw_path).replace('.tif', '_aligned.tif')
    aligned_path = os.path.join(os.path.dirname(dw_path), file_name)

    with rasterio.open(aligned_path, 'w',
                       driver='GTiff',
                       height=dst_height,
                       width=dst_width,
                       count=count,
                       dtype=source_dtype,
                       crs=dst_crs,
                       transform=dst_transform) as dst:
        dst.write(destination)

    print(f"DW prediction aligned and saved to: {aligned_path}")
    return aligned_path

#*ADD CONSENSUS IMAGE BAND AS NEW BAND TO THE DW-IMAGE*
def stack_dw_with_consensus(dw_path, consensus_path):
    with rasterio.open(dw_path) as dw_src:
        dw_data = dw_src.read()
        dw_meta = dw_src.meta.copy()

    with rasterio.open(consensus_path) as cons_src:
        cons_data = cons_src.read(1).astype(np.int16)

    if (dw_src.crs != cons_src.crs or
        dw_src.transform != cons_src.transform or
        dw_src.width != cons_src.width or
        dw_src.height != cons_src.height):
        raise ValueError("CRS, transform, or shape mismatch between DW and consensus rasters.")

    cons_data -= 1  # Adjust label range
    stacked = np.vstack([dw_data, cons_data[np.newaxis, ...]])

    dw_meta.update(count=stacked.shape[0], dtype='int16')

    name_no_ext = os.path.splitext(os.path.basename(dw_path))[0]
    out_path = os.path.join(os.path.dirname(dw_path), f"{name_no_ext}_with_consensus.tif")

    with rasterio.open(out_path, 'w', **dw_meta) as dst:
        dst.write(stacked)

    print(f"Saved stacked raster to: {out_path}")
    return out_path

#*GENERATE CLASSIFICATION MATRIX WITH PRECISION AND RECALL FOR ALL CLASSES*
def generate_classification_report(input_raster_path):
    with rasterio.open(input_raster_path) as src:
        dw_mode = src.read(5)
        consensus = src.read(7)

    mask = (consensus >= 0) & (consensus <= 8) & (dw_mode >= 0) & (dw_mode <= 8)
    y_true = consensus[mask].flatten()
    y_pred = dw_mode[mask].flatten()

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(9))

    with np.errstate(divide='ignore', invalid='ignore'):
        recall = np.diag(cm) / np.sum(cm, axis=1)
        precision = np.diag(cm) / np.sum(cm, axis=0)

    recall = np.nan_to_num(recall, nan=0.0)
    precision = np.nan_to_num(precision, nan=0.0)

    df = pd.DataFrame(cm, index=DW_CLASSES, columns=DW_CLASSES)
    df["Recall/User's (%)"] = np.round(recall * 100, 2)
    df.loc["Precision/Producer's (%)"] = list(np.round(precision * 100, 2)) + [np.nan]

    df.index.name = "Dynamic World (Predicted)"
    df.columns.name = "Expert Consensus (Ground Truth)"

    dw_dir = os.path.abspath(os.path.join(input_raster_path, os.pardir, os.pardir))
    output_dir = os.path.join(dw_dir, "DW_classification_results")
    os.makedirs(output_dir, exist_ok=True)

    output_csv_path = os.path.join(output_dir, "DW_classification_report_month.csv")
    df.to_csv(output_csv_path)
    print(f"Saved classification report to:\n{output_csv_path}")
