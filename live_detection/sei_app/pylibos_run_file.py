#!/usr/bin/python3
import argparse
import numpy as np
import json
import os
import sys
import torch
import math
import datetime
import mmap
import time
import copy
from collate_filter import *

def dsprint(level, msg):
    print(f"[{level}] - {msg}")

try:
    from pyomnisig import WidebandClassifier, WidebandClassifierMetadata, Manager, LicenseBuilder
    omnisig_available = True
except Exception as e:
    dsprint("ERROR", f"Importing OmniSIG: {e}")
    dsprint("ERROR", f"OmniSIG support will be disabled")
    omnisig_available = False


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafile', dest='datafile', default='samples.sigmf-data', type=str, help="data filename")
    parser.add_argument('--metafile-output', dest='metafile_output', default=None, type=str, help="file path to write meta, if not provided, will overwrite existing meta")
    parser.add_argument('--limit-frames', dest="limit_frames", help="exit after number of frames for testing", type=int, default=0)
    parser.add_argument('--enable-trt', help="Enable TRT processing (default=False)", action='store_true', required=False)
    parser.add_argument('--precision', help="trt precision", type=str, default="fp16", required=False)
    parser.add_argument('--runtime', help="Model file (runtime) to use (default=default model)", type=str, default="", required=False)
    parser.add_argument('--devices', help="Optional device list", type=str, default="", required=False)
    parser.add_argument('--license-path', help="Optional license file path", type=str, default="", required=False)
    parser.add_argument('--threshold',dest="threshold", type=float, default=0.75, help="Confidence threshold (between 0.0 and 1.0, defaults to 0.75)")
    parser.add_argument('--rssi-threshold',dest="rssi_threshold", type=float, default=None, help="RSSI threshold (any real number, defaults to disabled)")
    parser.add_argument('--unknown-threshold', dest="unknown_threshold", type=float, default=None, help="Unknown confidence threshold (between 0.0 and 1.0)")
    parser.add_argument('--collate', dest="enable_collate", help="Enable Collate Filter, default: False", action='store_true', required=False)

    args = parser.parse_args()
    return args

def get_samples_mmap(filename, datatype, offset, sample_count):
    if datatype == "cf32_le":
        start_byte = offset * 8  
        num_bytes = sample_count * 8

        with open(filename, "rb") as f:
          with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
            mm.seek(start_byte)
            raw_data = np.frombuffer(mm.read(num_bytes), dtype=np.complex64)
            data = raw_data.astype("complex64")

    elif datatype == "ci16_le":
        start_byte = offset * 4  
        num_bytes = sample_count * 4

        with open(filename, "rb") as f:
          with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
            mm.seek(start_byte)
            raw_data = np.frombuffer(mm.read(num_bytes), dtype=np.int16)
            data = raw_data[0::2] / 32767.0 + 1j * raw_data[1::2] / 32767.0
            data = data.astype("complex64")
    return data

def numpy_to_tensor(c64samples,  sample_count,  batch_size=1):
    sample_view = c64samples.view(np.float32).reshape(batch_size, sample_count, 2)
    sample_tensor = torch.from_numpy(sample_view)
    return sample_tensor

def write_json_to_file(filename, obj):
    with open(filename, 'w') as file:
        json.dump(obj, file, indent=2)

def get_meta(filename):
    try:
        return json.loads(open(filename).read())
    except json.decoder.JSONDecodeError:
        dsprint("ERROR", f"JSON error loading: {filename}")
        return None

def to_sigmf(anno):
    return_anno = {}
    return_anno['core:sample_count'] = anno.sample_count
    return_anno['core:sample_start'] = anno.start_sample
    return_anno['core:freq_lower_edge'] = anno.lower_freq
    return_anno['core:freq_upper_edge'] = anno.upper_freq
    return_anno['core:label'] = anno.description
    return_anno['deepsig:global_index'] = anno.global_index
    return_anno['deepsig:confidence'] = anno.confidence
    return_anno['deepsig:rssi'] = anno.rssi
    return_anno['deepsig:snr_estimate'] = anno.approx_snr
    return_anno['deepsig:full_seconds'] = anno.full_seconds
    return_anno['deepsig:fractional_seconds'] = anno.fractional_seconds
    return_anno['deepsig:sequence_number'] = anno.sequence_number

    if anno.AZ is not None or anno.EL is not None:
        return_anno['sigmf:signal_bearing'] = {
            'azimuth': anno.AZ, 
            'elevation': anno.EL
        }
    return return_anno

# def ts_to_full_frac(timestamp_str):
#     try:
#         dt_naive = datetime.datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%SZ")
#         dt_utc = dt_naive.replace(tzinfo=datetime.timezone.utc)
#     except ValueError:
#         dt_aware = datetime.datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S%z")
#         dt_utc = dt_aware.astimezone(datetime.timezone.utc)
#     epoch_float = dt_utc.timestamp()
#     epoch_int = int(epoch_float)
#     epoch_frac = epoch_float - epoch_int
#     return epoch_int, epoch_frac

def ts_to_full_frac(timestamp_str):
    formats = [
        ("%Y-%m-%dT%H:%M:%SZ", False),  # naive UTC format
        ("%Y-%m-%dT%H:%M:%S%z", True),    # timezone-aware format
        ("%Y-%m-%d-%H-%M-%S", False)      # alternative naive format
    ]
    
    dt = None
    for fmt, is_aware in formats:
        try:
            dt = datetime.datetime.strptime(timestamp_str, fmt)
            if is_aware:
                dt = dt.astimezone(datetime.timezone.utc)
            else:
                dt = dt.replace(tzinfo=datetime.timezone.utc)
            break  # Exit loop if parsing is successful
        except ValueError:
            continue
    
    if dt is None:
        raise ValueError("timestamp_str did not match any expected format")
    
    epoch_float = dt.timestamp()
    epoch_int = int(epoch_float)
    epoch_frac = epoch_float - epoch_int
    return epoch_int, epoch_frac

def epoch_to_full_frac(epoch_ts):
    epoch_int = int(epoch_ts)
    epoch_frac = epoch_ts - epoch_int
    return epoch_int, epoch_frac

def check_file(filename):
    if os.path.isfile(filename):
        dsprint("INFO", f"File found: {filename}")
    else:
        dsprint("ERROR", f"File not found: {filename}")
        sys.exit(1)

def merge_annotations2(annotation_list,  anno_merge_time=16, label_filter=None, cf_tolerance=0.05, bw_tolerance=0.1):
    # use a while loop because we will be removing annotations from this list as we merge them
    ii = 0
    removed_annotations = 0

    ann_list = copy.copy(annotation_list)
    
    if label_filter is not None and type(label_filter) == str:
        # Convert to standardized list format
        label_filter = [label_filter]
        
    updated_annotations = []
    
    while ii < len(ann_list):
        anno = ann_list[ii]
        start = anno['core:sample_start']
        count = anno['core:sample_count']
        end = start + count

        ### 2. Compare to Others until we pass the point annotations may be merged
        p_anno_idx = ii + 1
        if p_anno_idx >= len(ann_list):
            updated_annotations.append(dict(anno))
            break
        p_anno = ann_list[p_anno_idx]
        p_anno_start = p_anno['core:sample_start']
        
        # do not merge annotations with different labels
        if p_anno['core:label'] != anno['core:label'] or (label_filter is not None and anno['core:label'] not in label_filter):
            updated_annotations.append(dict(anno))
            ii += 1
            continue
        
        while p_anno_start <= (end + anno_merge_time):
            fc = (anno['core:freq_upper_edge'] + anno['core:freq_lower_edge']) / 2
            bw = anno['core:freq_upper_edge'] - anno['core:freq_lower_edge']
            
            p_fc = (p_anno['core:freq_upper_edge'] + p_anno['core:freq_lower_edge']) / 2
            p_bw = p_anno['core:freq_upper_edge'] - p_anno['core:freq_lower_edge']
            
            # percentage differences should be calculated relative to the bandwidth of the annotation since other metrics are arbitrary:
            fc_diff = abs(fc - p_fc) / bw
            bw_diff = abs(bw - p_bw) / bw
            
            if fc_diff < cf_tolerance and bw_diff < bw_tolerance:
                # this annotation should be merged and removed from the sorted list
                end = p_anno['core:sample_start'] + p_anno['core:sample_count']
                count = end - start
                anno['core:sample_count'] = count
                anno['core:freq_upper_edge'] = max(anno['core:freq_upper_edge'], p_anno['core:freq_upper_edge'])
                anno['core:freq_lower_edge'] = min(anno['core:freq_lower_edge'], p_anno['core:freq_lower_edge'])
                if 'deepsig:rssi' in anno:
                    anno['deepsig:rssi'] = max(anno['deepsig:rssi'], p_anno['deepsig:rssi'])
                    anno['deepsig:confidence'] = max(anno['deepsig:confidence'], p_anno['deepsig:confidence'])
                    anno['deepsig:snr_estimate'] = max(anno['deepsig:snr_estimate'], p_anno['deepsig:snr_estimate'])
                ann_list.remove(p_anno)
                removed_annotations += 1
            else:
                # if the annotation was not merged, go on to the next index
                p_anno_idx += 1

            if p_anno_idx >= len(ann_list):
                updated_annotations.append(dict(anno))
                break
            p_anno = ann_list[p_anno_idx]
            p_anno_start = p_anno['core:sample_start']
        
        # annotation is now merged and can be added to the table
        updated_annotations.append(dict(anno))
        ii += 1
        
    return updated_annotations

def merge_annotations(annotation_list, anno_merge_time=16, label_filter=None, cf_tolerance=0.05, bw_tolerance=0.1):
    """
    Merges annotations if they:
      - Have the same label (and pass optional label_filter).
      - Overlap in time OR start within anno_merge_time of each other.
      - Have center-frequency / bandwidth differences within cf_tolerance/bw_tolerance.
    Returns a list of merged annotations.
    """

    # If label_filter is a single string, make it a list
    if label_filter is not None and isinstance(label_filter, str):
        label_filter = [label_filter]

    # Sort the annotations by their start time so merging is consistent
    ann_list = sorted(annotation_list, key=lambda x: x['core:sample_start'])
    ann_list = copy.copy(ann_list)  # so we can remove items as we go

    updated_annotations = []
    i = 0

    while i < len(ann_list):
        anno = ann_list[i]
        label = anno['core:label']
        start = anno['core:sample_start']
        end = start + anno['core:sample_count']

        # If we have a label_filter and the current label isn't in it, no merging attempt
        if label_filter is not None and label not in label_filter:
            updated_annotations.append(dict(anno))
            i += 1
            continue

        j = i + 1
        while j < len(ann_list):
            p_anno = ann_list[j]
            p_label = p_anno['core:label']
            p_start = p_anno['core:sample_start']
            p_end = p_start + p_anno['core:sample_count']

            # If p_anno is too far in time, no point checking further
            # (It starts after end + anno_merge_time)
            if p_start > end + anno_merge_time:
                break

            # If labels differ (or fail filter), skip this candidate
            if p_label != label:
                j += 1
                continue
            if label_filter is not None and p_label not in label_filter:
                j += 1
                continue

            # Check frequency overlap
            fc = (anno['core:freq_upper_edge'] + anno['core:freq_lower_edge']) / 2
            bw = anno['core:freq_upper_edge'] - anno['core:freq_lower_edge']
            p_fc = (p_anno['core:freq_upper_edge'] + p_anno['core:freq_lower_edge']) / 2
            p_bw = p_anno['core:freq_upper_edge'] - p_anno['core:freq_lower_edge']

            # To avoid divide-by-zero if bw = 0, we handle that case gracefully
            if bw == 0:
                fc_diff = 0
                bw_diff = 0
            else:
                fc_diff = abs(fc - p_fc) / bw
                bw_diff = abs(bw - p_bw) / bw

            # Merge if frequency differences are within tolerance
            if fc_diff < cf_tolerance and bw_diff < bw_tolerance:
                # Extend our "end" if p_anno goes further
                if p_end > end:
                    end = p_end
                    anno['core:sample_count'] = end - start

                # Update frequency edges
                anno['core:freq_upper_edge'] = max(anno['core:freq_upper_edge'], p_anno['core:freq_upper_edge'])
                anno['core:freq_lower_edge'] = min(anno['core:freq_lower_edge'], p_anno['core:freq_lower_edge'])

                # Merge any optional fields (only if present in both)
                if 'deepsig:rssi' in anno and 'deepsig:rssi' in p_anno:
                    anno['deepsig:rssi'] = max(anno['deepsig:rssi'], p_anno['deepsig:rssi'])
                if 'deepsig:confidence' in anno and 'deepsig:confidence' in p_anno:
                    anno['deepsig:confidence'] = max(anno['deepsig:confidence'], p_anno['deepsig:confidence'])
                if 'deepsig:snr_estimate' in anno and 'deepsig:snr_estimate' in p_anno:
                    anno['deepsig:snr_estimate'] = max(anno['deepsig:snr_estimate'], p_anno['deepsig:snr_estimate'])

                # Remove the merged annotation from the list
                ann_list.pop(j)
            else:
                # Not mergeable by frequency or bandwidth, go to next candidate
                j += 1

        # After checking merges, push the final merged annotation
        updated_annotations.append(dict(anno))
        i += 1

    return updated_annotations

def main():
    if not omnisig_available:
        dsprint("ERROR", f"Unable to load pyomnisig.  Exiting.")
        sys.exit(1)

    args = get_args()

    datafile = args.datafile
    metafile = datafile.replace("sigmf-data", "sigmf-meta")

    check_file(datafile)
    check_file(metafile)
    
    md = get_meta(metafile)
    meta_sr = int(md['global']['core:sample_rate'])
    meta_dt = md['global']['core:datatype']
    meta_cf = int(md['captures'][0]['core:frequency'])
    meta_st = md['captures'][0]['core:datetime']
    meta_datasize = 8 if meta_dt == "cf32_le" else 4
    number_of_samples = int(os.path.getsize(datafile)/meta_datasize)
    dsprint("INFO", f"File Length: {number_of_samples/meta_sr} seconds")

    if args.enable_trt:
        precision = args.precision
        dsprint("INFO", f"Enabling TRT")
    else:
        precision = "fp32"
    dsprint("INFO", f"Precision: {precision}")
    
    try:
        manager = Manager.FromLicenseBuilder(LicenseBuilder())
        generator = WidebandClassifier(
            manager,
            meta_sr,
            1,   # thread_profiles
            1,   # batch_size
            1,   # channels
            0,   # inference_channel
            precision,
            args.devices,
            args.runtime,
            args.enable_trt,
        )
    except Exception as e:
        dsprint("ERROR", f"Creating omnisig classifier: {e}")
        sys.exit(2)
        
    samples_per_inference = generator.get_samples_per_inference()
    dsprint("INFO", f"Samples per inference: {samples_per_inference}")

    num_complete_frames = math.floor(number_of_samples/samples_per_inference)
    dsprint("INFO", f"Number of complete frames: {num_complete_frames}")
    
    ts_start_full, ts_start_frac = ts_to_full_frac(meta_st)
    filetime = ts_start_full + ts_start_frac

    dsprint("INFO", f"File datetime: {meta_st}")
    out_ants = []
    ct = 0
    global_index = 0
    sample_offset = 0

    if args.enable_collate:
        dsprint("INFO", f"Enabling Collate Filter")
        collate_filter = CollateFilter(streaming=True)

    t0 = time.time()
    for idx in range(0,num_complete_frames):
        chunk = get_samples_mmap(datafile, meta_dt, sample_offset, samples_per_inference)

        time_since_start = sample_offset / meta_sr
        run_ts = filetime + time_since_start
        ts_run_full, ts_run_frac = epoch_to_full_frac(run_ts)
        dsprint("INFO", f"Frame CT: {ct} Index: {global_index} Full TS: {ts_run_full} Frac TS: {ts_run_frac}")

        wcmd = WidebandClassifierMetadata(
                sample_rate=meta_sr,
                center_frequency=meta_cf,
                gain=1,
                sequence_number=0,
                number_samples=samples_per_inference,
                local_sample_start_index=global_index,
                global_sample_start_index=global_index,
                timestamp_pair=[ts_run_full, ts_run_frac]
        )
        
        sample_tensor = numpy_to_tensor(chunk, samples_per_inference)
        result = generator.infer([sample_tensor], wcmd)
        
        if args.enable_collate:
            for frame_result in result:
                filtered_result = collate_filter.apply(frame_result,  wcmd)
                    
                for frame_annotation in collate_filter.get_filtered_annotations():
                    ant = to_sigmf(frame_annotation)
                    out_ants.append(ant)
        else:
            for frame_result in result:
                for frame_annotation in frame_result.annotations:
                    ant = to_sigmf(frame_annotation)
                    out_ants.append(ant)

        global_index += samples_per_inference
        sample_offset += samples_per_inference
        ct += 1

        if args.limit_frames > 0:
            if ct >= args.limit_frames:
                break
    t1 = time.time()
    print(f"run time: {t1-t0}")
    print(f"out_ants len: {len(out_ants)}")
    
    out_ants = merge_annotations(out_ants, 100, None, 0.25, 0.5)
    print(f"out_ants len post merge: {len(out_ants)}")

    if args.unknown_threshold is not None:
        dsprint("INFO", f"Unknown Confidence: {args.unknown_threshold}")
        for ant in out_ants:
            if ant['deepsig:confidence'] < args.unknown_threshold:
                ant["core:label"] = "Unknown"

    dsprint("INFO", f"Filtering Confidence: {args.threshold}")
    out_ants = [ant for ant in out_ants if ant['deepsig:confidence'] >= args.threshold]

    if args.rssi_threshold is not None:
        dsprint("INFO", f"Filtering RSSI: {args.rssi_threshold}")
        out_ants = [ant for ant in out_ants if ant['deepsig:rssi'] >= args.rssi_threshold]

    

    md['annotations'] = out_ants

    if args.metafile_output is not None:
        mf_to_write = args.metafile_output
    else:
        mf_to_write = metafile

    dsprint("INFO", f"Writing metadata to: {mf_to_write}")
    write_json_to_file(mf_to_write, md)

if __name__ == "__main__":
    timestart = time.time()
    main()
    timeend = time.time()
    tdiff = timeend - timestart
    print(f"time: {tdiff}")