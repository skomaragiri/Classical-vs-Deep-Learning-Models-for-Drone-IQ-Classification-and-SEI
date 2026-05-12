import json
import sys
import mmap
import numpy as np
import torch
import datetime
import os

def dsprint(level, msg):
    print(f"[{level}] - {msg}")

def write_json_to_file(filename, obj):
    with open(filename, 'w') as file:
        json.dump(obj, file, indent=2)

def get_meta(filename):
    try:
        return json.loads(open(filename).read())
    except json.decoder.JSONDecodeError:
        dsprint("ERROR", f"JSON error loading: {filename}")
        return None

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

    #  direction-finder
    if anno.AZ is not None or anno.EL is not None:
        return_anno['sigmf:signal_bearing'] = {
            'azimuth': anno.AZ, 
            'elevation': anno.EL
        }

    return return_anno

# def ts_to_full_frac(timestamp_str):
#     dt_naive = datetime.datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%SZ")
#     dt_utc = dt_naive.replace(tzinfo=datetime.timezone.utc)
#     epoch_float = dt_utc.timestamp()
#     epoch_int = int(epoch_float)
#     epoch_frac = epoch_float - epoch_int
#     return epoch_int, epoch_frac

def ts_to_full_frac(timestamp_str):
    try:
        dt_naive = datetime.datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%SZ")
        dt_utc = dt_naive.replace(tzinfo=datetime.timezone.utc)
    except ValueError:
        dt_aware = datetime.datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S%z")
        dt_utc = dt_aware.astimezone(datetime.timezone.utc)
    epoch_float = dt_utc.timestamp()
    epoch_int = int(epoch_float)
    epoch_frac = epoch_float - epoch_int
    return epoch_int, epoch_frac
    
def epoch_to_full_frac(epoch_ts):
    epoch_int = int(epoch_ts)
    epoch_frac = epoch_ts - epoch_int
    return epoch_int, epoch_frac

def check_file(dslog, filename):
    if os.path.isfile(filename):
        dslog.info(f"File found: {filename}")
    else:
        dslog.error(f"File not found: {filename}")
        sys.exit(1)