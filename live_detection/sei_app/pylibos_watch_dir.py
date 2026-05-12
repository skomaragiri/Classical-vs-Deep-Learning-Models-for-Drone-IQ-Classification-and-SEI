#!/usr/bin/python3
import argparse
import numpy as np
import json
import os
import sys
import torch
import math
import datetime
import threading
from queue import Queue, Empty
import shutil
from pathlib import Path
import time
import mmap
import zmq
from collate_filter import *
import copy

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from dslogger import Logger
from pylibos_common import *

from pyomnisig import WidebandClassifier, WidebandClassifierMetadata

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--watchdir', dest='watchdir', default=None, type=str, help="dir to watch for sigmf writes", required=True)
    parser.add_argument('--output-path', dest='output_path', default=None, type=str, help="dir to output processed files", required=True)
    parser.add_argument('--sample-rate', dest='sample_rate', default=125e6, type=float, help="sampling rate (default 125e6 to match the OmniSIG engine model)")
    parser.add_argument('--enable-trt', '-t', help="Enable TRT processing (default=False)", action='store_true', required=False)
    parser.add_argument('--precision', help="trt precision", type=str, default="fp16", required=False)
    parser.add_argument('--runtime', '-r', help="Model file (runtime) to use (default=default model)", type=str, default="", required=False)
    parser.add_argument('--devices', help="Optional device list", type=str, default="", required=False)
    parser.add_argument('--license-path', '-l', help="Optional license file path", type=str, default="", required=False)
    parser.add_argument('--verbose', dest='verbose', action='store_true')
    parser.add_argument('--log-level', type=str, default='DEBUG', choices=['debug', 'info', 'warning', 'error', 'critical'],help='Set the logging level')
    parser.add_argument('--write-log', dest='write_log', action='store_true')
    parser.add_argument('--threshold',dest="threshold", type=float, default=0.75, help="Confidence threshold (between 0.0 and 1.0, defaults to 0.75)")
    parser.add_argument('--rssi-threshold',dest="rssi_threshold", type=float, default=None, help="RSSI threshold (any real number, defaults to disabled)")
    parser.add_argument('--unknown-threshold',dest="unknown_threshold", type=float, default=None, help="Unknown confidence threshold (between 0.0 and 1.0)")
    parser.add_argument('--collate', dest="enable_collate", help="Enable Collate Filter, default: False", action='store_true', required=False)

    parser.add_argument('--enable-zmq', dest='enable_zmq', action='store_true')
    parser.add_argument("--port", default="9100", help="Port to bind the ZMQ PUB socket on")
    args = parser.parse_args()
    return args

def omnisig_generator(dslog, args):
    if args.enable_trt:
        precision = args.precision
        dslog.debug(f"Enabling TRT")
    else:
        precision = "fp32"
    dslog.debug(f"Precision: {precision}")

    try:
        from pyomnisig import Manager
        manager = Manager()
        generator = WidebandClassifier(
            manager,
            sample_rate=args.sample_rate,
            thread_profiles=1,
            batch_size=1,
            channels=1,
            inference_channel=0,
            precision=precision,
            devices=args.devices,
            runtime=args.runtime,
            trt=args.enable_trt,
        )
        return generator
    except Exception as e:
        dslog.error(f"Creating omnisig classifier: {e}")
        sys.exit(2)

def run_omnisig(dslog, datafile, generator, zmq_socket, args):
    metafile = datafile.replace("sigmf-data", "sigmf-meta")
    check_file(dslog, datafile)
    check_file(dslog, metafile)

    md = get_meta(metafile)
    meta_sr = int(md['global']['core:sample_rate'])
    meta_dt = md['global']['core:datatype']
    meta_cf = int(md['captures'][0]['core:frequency'])
    meta_st = md['captures'][0]['core:datetime']
    meta_datasize = 8 if meta_dt == "cf32_le" else 4
    number_of_samples = int(os.path.getsize(datafile)/meta_datasize)
    dslog.info(f"File Length: {number_of_samples/meta_sr} seconds")
    
    samples_per_inference = generator.get_samples_per_inference()
    dslog.info(f"Samples per inference: {samples_per_inference}")

    num_complete_frames = math.floor(number_of_samples/samples_per_inference)
    dslog.info(f"Number of complete frames: {num_complete_frames}")

    ts_start_full, ts_start_frac = ts_to_full_frac(meta_st)
    filetime = ts_start_full + ts_start_frac

    dslog.info(f"File datetime: {meta_st}")
    out_ants = []
    ct = 0
    global_index = 0
    sample_offset = 0

    if args.enable_collate:
        dslog.info(f"Enabling Collate Filter")
        collate_filter = CollateFilter(streaming=True)
        collate_filter.set_absolute_absence_threshold(1.0)
        collate_filter.set_relative_absence_threshold(0.5)
        
    for idx in range(0,num_complete_frames):
        chunk = get_samples_mmap(datafile, meta_dt, sample_offset, samples_per_inference)

        time_since_start = sample_offset / meta_sr
        run_ts = filetime + time_since_start
        ts_run_full, ts_run_frac = epoch_to_full_frac(run_ts)
        dslog.info(f"Frame CT: {ct} Index: {global_index} Full TS: {ts_run_full} Frac TS: {ts_run_frac}")

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
                frame_result = collate_filter.apply(frame_result,  wcmd)
                    
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

    # out_ants = merge_annotations(out_ants, 100, None, 0.25, 0.5)
    # print(f"out_ants len post merge: {len(out_ants)}")

    if args.unknown_threshold is not None:
        dslog.info(f"Unknown Confidence: {args.unknown_threshold}")
        for ant in out_ants:
            if ant['deepsig:confidence'] < args.unknown_threshold:
                ant["core:label"] = "Unknown"

    dslog.info(f"Filtering Confidence: {args.threshold}")
    out_ants = [ant for ant in out_ants if ant['deepsig:confidence'] >= args.threshold]

    if args.rssi_threshold is not None:
        dslog.info(f"Filtering RSSI: {args.rssi_threshold}")
        out_ants = [ant for ant in out_ants if ant['deepsig:rssi'] >= args.rssi_threshold]

    md['annotations'] = out_ants

    metadata_filename_only = os.path.basename(metafile)
    new_metafile_name = os.path.join(args.output_path, metadata_filename_only)

    datafile_filename_only = os.path.basename(datafile)
    new_datafile_name = os.path.join(args.output_path, datafile_filename_only)
    dslog.info(f"Moving datafile from: {datafile} to: {new_datafile_name}")
    shutil.move(datafile, new_datafile_name)

    dslog.info(f"Writing metadata to: {new_metafile_name}")
    write_json_to_file(new_metafile_name, md)

    dslog.info(f"Deleting original metadata: {metafile}")
    os.remove(metafile)

    if args.enable_zmq and zmq_socket is not None:
        dslog.info(f"Sending SigMF over ZMQ")
        zmq_socket.send_json(md)
        time.sleep(.1)

    dslog.info(f"Done.")

class SigMFFileHandler(FileSystemEventHandler):
    def __init__(self, dslog, job_queue):
        super().__init__()
        self.dslog = dslog
        self.job_queue = job_queue
        self.pending = {}

    def on_created(self, event):
        if event.is_directory:
            return 

        filepath = Path(event.src_path)
        suffix = filepath.suffix 

        if suffix in [".sigmf-meta", ".sigmf-data"]:
            prefix = filepath.with_suffix('')

            if prefix not in self.pending:
                self.pending[prefix] = {"meta": None, "data": None}

            if suffix == ".sigmf-meta":
                self.pending[prefix]["meta"] = filepath
            elif suffix == ".sigmf-data":
                self.pending[prefix]["data"] = filepath

            if (self.pending[prefix]["meta"] is not None and
                self.pending[prefix]["data"] is not None):
                meta_path = self.pending[prefix]["meta"]
                data_path = self.pending[prefix]["data"]
                
                self.dslog.info(f"[Watcher] Found complete pair:")
                self.dslog.info(f"  Meta: {meta_path}")
                self.dslog.info(f"  Data: {data_path}")
                
                time.sleep(1) # fixes a race condition 
                self.job_queue.put(str(data_path))

                del self.pending[prefix]

def watch_directory(dslog, job_queue, stop_event, args):
    event_handler = SigMFFileHandler(dslog, job_queue)
    observer = Observer()
    observer.schedule(event_handler, args.watchdir, recursive=False)
    observer.start()

    dslog.info(f"[Watcher] Started watching: {args.watchdir}")
    try:
        while not stop_event.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass

    dslog.info("[Watcher] Stopping...")
    observer.stop()
    observer.join()
    dslog.info("[Watcher] Stopped")

def process_data_files(dslog, job_queue, stop_event, generator, zmq_socket, args):
    dslog.info("[Worker] Started worker thread for processing .sigmf-data")
    while True:
        if stop_event.is_set():
            break

        try:
            data_file = job_queue.get(timeout=0.5)
        except Empty:
            continue

        dslog.info(f"[Worker] Processing file: {data_file}")
        run_omnisig(dslog, data_file, generator, zmq_socket, args)
        dslog.info(f"[Worker] Done processing: {data_file}")
        job_queue.task_done()
    dslog.info("[Worker] Exiting processing thread")

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
    args = get_args()

    dslog_instance = Logger(args.verbose, args.log_level.upper(), args.write_log)
    dslog = dslog_instance.get_logger()
    dslog.info(f"Logging Level: {args.log_level}")
    dslog.info(f"Logging to memory...")
    dslog.info(f"Watching dir: {args.watchdir}")

    if args.enable_zmq:
        zmq_endpoint = f"tcp://*:{args.port}"
        dslog.info(f"Enabling ZMQ: {zmq_endpoint}")
        context = zmq.Context()
        zmq_socket = context.socket(zmq.PUB)
        zmq_socket.bind(zmq_endpoint)
    else:
        zmq_socket = None

    job_queue = Queue()
    stop_event = threading.Event()

    generator = omnisig_generator(dslog, args)

    watcher_thread = threading.Thread(
        target=watch_directory,
        args=(dslog, job_queue, stop_event, args),
        daemon=True,
        name="watch_directory"
    )

    worker_thread = threading.Thread(
        target=process_data_files,
        args=(dslog, job_queue, stop_event, generator, zmq_socket, args),
        daemon=True,
        name="process_data_files"
    )

    watcher_thread.start()
    worker_thread.start()

    dslog.info("[Main] Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        dslog.info("[Main] Keyboard interrupt received. Shutting down...")

    stop_event.set()

    watcher_thread.join()
    worker_thread.join()

    dslog.info("[Main] All done.")

if __name__ == "__main__":
    main()