#!/usr/bin/python3

# ------------  Collate Filter ----------------------------------------
class CollatedAccumulator(object):
    def __init__(self, value, weight=1):
        self.total_weight = 0
        self.weighted_sum = 0
        self.weighted_square_sum = 0
        self.max_val = float('-inf')
        self.min_val = float('inf')
        self.count = 0
        
        self.add(value, weight)

    def add(self, value, weight=1):
        self.total_weight += weight
        self.weighted_sum += value * weight
        self.weighted_square_sum += (value**2) * weight
        self.max_val = max(self.max_val, value)
        self.min_val = min(self.min_val, value)
        self.count += 1

    def weighted_mean(self):
        return self.weighted_sum / self.total_weight if self.total_weight != 0 else 0

    def weighted_variance(self):
        if self.total_weight == 0 or self.count == 1:
            return 0
        mean = self.weighted_mean
        variance = (self.weighted_square_sum / self.total_weight) - (mean**2)
        return variance

    def sum_of_weights(self):
        return self.total_weight

    def max(self):
        return self.max_val

    def min(self):
        return self.min_val

    def total_count(self):
        return self.count

class collated_annotation(object):
    def __init__(self, ann, sequence_index):
        self.annotation = ann

        self.sequence_index = sequence_index
        self.description_histogram = {}
        self.azimuth_histogram = {}
        self.elevation_histogram = {}
        
        self.active_sample_count = self.annotation.sample_count
        self.local_sample_count = self.annotation.sample_count
        self.sequence_sample_count = self.annotation.sample_count
        self.description_histogram[self.annotation.description] = self.annotation.sample_count
        
        self.center_freq_accumulator = CollatedAccumulator(self.get_center_freq(), weight = self.annotation.sample_count)
        self.bandwidth_accumulator = CollatedAccumulator(self.get_bandwidth(), weight = self.annotation.sample_count)
        self.confidence_accumulator = CollatedAccumulator(self.annotation.confidence, weight = self.annotation.sample_count)
        self.approx_snr_accumulator = CollatedAccumulator(self.annotation.approx_snr, weight = self.annotation.sample_count)
        self.rssi_accumulator = CollatedAccumulator(self.annotation.rssi, weight = self.annotation.sample_count)
        
        #if (annotation.AZ is not None):
        #    self.azimuth_accumulator = CollatedAccumulator(annotation.AZ, weight = annotation.sample_count)
        #    self.azimuth_histogram[annotation.AZ] = annotation.sample_count

        #if (annotation.EL is not None):
        #    self.elevation_accumulator = CollatedAccumulator(annotation.EL, weight = annotation.sample_count)
        #    self.elevation_histogram[annotation.EL] = annotation.sample_count

    def get_annotation(self, expired=False):
        self.annotation.confidence = self.confidence_accumulator.weighted_mean()
        self.annotation.approx_snr = self.approx_snr_accumulator.weighted_mean()
        self.annotation.rssi = self.rssi_accumulator.weighted_mean()

        # self.extra_statistics.detection_count = self.center_freq_accumulator.count()
        # self.extra_statistics.active_sample_count = self.center_freq_accumulator.sum_of_weights()
        # self.extra_statistics.center_freq_std = np.sqrt(self.center_freq_accumulator.weighted_variance())
        # self.extra_statistics.bandwidth_std = np.sqrt(self.bandwidth_accumulator.weighted_variance())
        # self.extra_statistics.confidence_std = np.sqrt(self.confidence_accumulator.weighted_variance())
        # self.extra_statistics.approx_snr_std = np.sqrt(self.approx_snr_accumulator.weighted_variance())
        # self.extra_statistics.rssi_std = np.sqrt(self.rssi_accumulator.weighted_variance())
        # self.extra_statistics.center_freq_min = self.center_freq_accumulator.min()
        # self.extra_statistics.bandwidth_min = self.bandwidth_accumulator.min()
        # self.extra_statistics.confidence_min = self.confidence_accumulator.min()
        # self.extra_statistics.approx_snr_min = self.approx_snr_accumulator.min()
        # self.extra_statistics.rssi_min = self.rssi_accumulator.min()
        # self.extra_statistics.center_freq_max = self.center_freq_accumulator.max()
        # self.extra_statistics.bandwidth_max = self.bandwidth_accumulator.max()
        # self.extra_statistics.confidence_max = self.confidence_accumulator.max()
        # self.extra_statistics.approx_snr_max = self.approx_snr_accumulator.max()
        # self.extra_statistics.rssi_max = self.rssi_accumulator.max()

        #if (!_azimuth_histogram.empty()) {
        #    _annotation.AZ = acc::weighted_mean(_azimuth_accumulator);
        #    _annotation.extra_statistics->azimuth_std = std::sqrt(acc::weighted_variance(_azimuth_accumulator));
        #    _annotation.extra_statistics->azimuth_min = acc::min(_azimuth_accumulator);
        #    _annotation.extra_statistics->azimuth_max = acc::max(_azimuth_accumulator);
        #    _annotation.extra_statistics->azimuth_histogram = _azimuth_histogram;
        #}

        #if (!_elevation_histogram.empty()) {
        #    _annotation.EL = acc::weighted_mean(_elevation_accumulator);
        #    _annotation.extra_statistics->elevation_std = std::sqrt(acc::weighted_variance(_elevation_accumulator));
        #    _annotation.extra_statistics->elevation_min = acc::min(_elevation_accumulator);
        #    _annotation.extra_statistics->elevation_max = acc::max(_elevation_accumulator);
        #    _annotation.extra_statistics->elevation_histogram = _elevation_histogram;
        #}

        self.annotation_expired = expired
        return self.annotation

    def get_center_freq(self):
        return 0.5 * (self.annotation.upper_freq + self.annotation.lower_freq)

    def get_bandwidth(self):
        return self.annotation.upper_freq - self.annotation.lower_freq

    def get_local_end_sample(self):
        return self.annotation.start_sample + self.local_sample_count

    def get_global_end_sample(self):
        return self.annotation.global_index + self.annotation.sample_count

    def get_sequence_end_sample(self):
        return self.sequence_index + self.sequence_sample_count

    def get_freq_offset(self, other):
        return abs(self.get_center_freq() - other.get_center_freq()) / self.get_bandwidth()

    def get_bandwidth_ratio(self, other):
        bandwidth = self.get_bandwidth()
        other_bandwidth = other.get_bandwidth()
        return min(bandwidth, other_bandwidth) / max(bandwidth, other_bandwidth)

    def get_add_local_sample_count(self, other):
        end_sample = self.get_local_end_sample()
        other_end_sample = other.get_local_end_sample()
        return max(other_end_sample, end_sample) - end_sample

    def get_add_global_sample_count(self, other):
        end_sample = self.get_global_end_sample()
        other_end_sample = other.get_global_end_sample()
        return max(other_end_sample, end_sample) - end_sample

    def get_add_sequence_sample_count(self, other):
        end_sample = self.get_sequence_end_sample()
        other_end_sample = other.get_sequence_end_sample()
        return max(other_end_sample, end_sample) - end_sample

    def get_add_active_sample_count(self, other):
        end_sample = self.get_sequence_end_sample()
        other_end_sample = other.get_sequence_end_sample()
        return max(other_end_sample, end_sample) - max(other.sequence_index, end_sample)

    def get_absolute_absence_other(self, other):
        return self.get_add_sequence_sample_count(other) - self.get_add_active_sample_count(other)

    def get_absolute_absence(self, cursor):
        end_sample = self.get_sequence_end_sample()
        return max(cursor, end_sample) - end_sample

    def get_relative_absence_other(self, other):
        sample_count = self.sequence_sample_count + self.get_add_sequence_sample_count(other)
        active_sample_count = self.active_sample_count + self.get_add_active_sample_count(other)
        return 1.0 - float(active_sample_count) / float(sample_count)

    def get_relative_absence(self, cursor):
        sample_count = self.sequence_sample_count + self.get_absolute_absence(cursor)
        return 1.0 - float(self.active_sample_count) / float(sample_count)

    def get_active_sample_count(self):
        return self.active_sample_count

    def extend(self, other, alpha=1.0):
        self.active_sample_count += self.get_add_active_sample_count(other)
        self.local_sample_count += self.get_add_local_sample_count(other)
        self.sequence_sample_count += self.get_add_sequence_sample_count(other)
        self.annotation.sample_count += self.get_add_global_sample_count(other)

        if (alpha != 1.0):
            for cur_key in self.description_histogram.keys():
                self.description_histogram[cur_key] *= alpha

        #if other.annotation.demod_info is not None and len(other.annotation.demod_info) > 0:
        #    self.annotation.demod_info = other.annotation.demod_info

        try:
            self.description_histogram[other.annotation.description] += other.center_freq_accumulator.sum_of_weights()
        except Exception as e:
            print("Collate Extend() error in description_histogram[description]: Label mismatch: " + str(e))
            return
            
        self.center_freq_accumulator.add(other.get_center_freq(), weight = other.center_freq_accumulator.sum_of_weights())
        self.bandwidth_accumulator.add(other.get_bandwidth(), weight = other.bandwidth_accumulator.sum_of_weights())
            
        max_description_count = 0

        for cur_key in self.description_histogram.keys():
            item = self.description_histogram[cur_key]
            
            if (item > max_description_count):
                try:
                    self.annotation.description = cur_key
                    max_description_count = item
                except Exception as e:
                    print("collate extend() error setting description: " + str(e))
            
        center_frequency = self.center_freq_accumulator.weighted_mean()
        bandwidth = self.bandwidth_accumulator.weighted_mean()
        self.annotation.lower_freq = center_frequency - 0.5 * bandwidth
        self.annotation.upper_freq = center_frequency + 0.5 * bandwidth

        self.confidence_accumulator.add(other.annotation.confidence, weight = other.confidence_accumulator.sum_of_weights())
        self.approx_snr_accumulator.add(other.annotation.approx_snr, weight = other.approx_snr_accumulator.sum_of_weights())
        self.rssi_accumulator.add(other.annotation.rssi, weight = other.rssi_accumulator.sum_of_weights())

        #if (other.annotation.AZ is not None):
        #    self.azimuth_accumulator.add(other.annotation.AZ, weight = other.azimuth_accumulator.sum_of_weights())
        #    for (const auto &[azimuth, sample_count] : other._azimuth_histogram):
        #        self.azimuth_histogram[azimuth] += sample_count

        #if (other.annotation.EL is not None):
        #    self.elevation_accumulator.add(other.annotation.EL, weight = other.azimuth_accumulator.sum_of_weights())
        #    for (const auto &[elevation, sample_count] : other._elevation_histogram):
        #        self.elevation_histogram[elevation] += sample_count

class CollateFilter(object):
    def __init__(self, streaming=False):
        self.in_annotation_count = 0
        self.out_annotation_count = 0
        self.frame_sequence_index = 0

        self.active_signals = []
        self.final_res_annotations = []
        
        self.streaming = streaming
        self.expirations = True

        self.iou_threshold = 0.8
        self.iou_tolerance_bins = 1.5
        # the maximum absolute time (in seconds) that a signal can be absent before the next annotation gets assigned to a new signal
        self.absolute_absence_threshold = 0.1
        # the maximum fraction of the total time that a signal can be absent before the next annotation gets assigned to a new signal
        self.relative_absence_threshold = 0.1
        
    def set_absolute_absence_threshold(self, threshold):
        self.absolute_absence_threshold = threshold
        
    def set_relative_absence_threshold(self, threshold):
        self.relative_absence_threshold = threshold
        
    def flush(self):
        annotations = []
        for sig in self.active_signals:
            annotations.append(sig.get_annotation())

        if not self.streaming:
            self.active_signals = []

        return annotations

    def fuzzy_iou(ann1_lower_edge, ann1_upper_edge, ann2_lower_edge, ann2_upper_edge, tolerance):
        offset = 0.5 * ((ann1_lower_edge + ann1_upper_edge) - (ann2_lower_edge + ann2_upper_edge))

        if (offset > 0):
            slew = min(tolerance, offset)
        else:
            slew = max(-tolerance, offset)

        ann1_upper_edge -= slew;
        ann1_lower_edge -= slew;

        i = max(0., min(ann2_upper_edge, ann1_upper_edge) - max(ann2_lower_edge, ann1_lower_edge))
        u = max(0., max(ann2_upper_edge, ann1_upper_edge) - min(ann2_lower_edge, ann1_lower_edge))
        return min(1., i / u)
    
    # def apply(WidebandClassifierResult &res, WidebandClassifierMetadata &md):
    def get_filtered_annotations(self):
        return self.final_res_annotations
        
    def apply(self,  res, md):
        iou_tolerance = self.iou_tolerance_bins * (md.sample_rate / float(res.spectrogram_nfft))

        # clear out the annotations in the inference result to be filled with collated annotations
        original_annotations = res.annotations
        self.final_res_annotations = []
        res.annotations = []

        # sort the annotations so that they are processed in order of start sample
        original_annotations = sorted(original_annotations, key=lambda d: d.start_sample)

        for ann in original_annotations:
            if (ann.sample_count == 0):
                # DS_LOG_DEBUG("Zero length annotation");
                continue

            if (ann.upper_freq <= ann.lower_freq):
                #DS_LOG_DEBUG("Zero bandwidth annotation");
                continue

            self.in_annotation_count += 1
            
            match_found = False
            
            sequence_index = self.frame_sequence_index + ann.start_sample - md.local_sample_start_index
            current_annotation = collated_annotation(ann, sequence_index)
            
            updates = []
            expirations = []
            matches = {}  # This will be a dict with double keys, and values will be a list.

            #  find all active signals that have close enough bandwidth and center frequency to the current annotation
            for iter in self.active_signals:
                iou = CollateFilter.fuzzy_iou(
                    current_annotation.get_center_freq() - 0.5 * current_annotation.get_bandwidth(),
                    current_annotation.get_center_freq() + 0.5 * current_annotation.get_bandwidth(),
                    iter.get_center_freq() - 0.5 * iter.get_bandwidth(),
                    iter.get_center_freq() + 0.5 * iter.get_bandwidth(),
                    iou_tolerance
                )
                if iou >= self.iou_threshold:
                    freq_offset = iter.get_freq_offset(current_annotation)
                    
                    if freq_offset in matches:
                        matches[freq_offset][current_annotation.annotation.description] = iter
                    else:
                        matches[freq_offset] = {current_annotation.annotation.description: iter }

            # Matches was originally a sorted dict in C++, so match that here:
            matchKeys = list(matches.keys())
            matchKeys.sort()
            matches = {i: matches[i] for i in matchKeys}

            # working from the closest in frequency, find the first matching signal which will not violate absence
            # thresholds if it is added to that signal
            for match_key in matches.keys():
                # In C++, matches is a multimap (like a dict, but keys don't have to be unique).  There's no C++ equivilent, so we'll 
                # Use a dict for each unique key to separate out labels.
                item_dict = matches[match_key]
                
                for cur_key in item_dict.keys():
                    item = item_dict[cur_key]
                    
                    if item.annotation.description != current_annotation.annotation.description:
                        continue
                        
                    absolute_absence = item.get_absolute_absence_other(current_annotation)
                    relative_absence = item.get_relative_absence_other(current_annotation)

                    # check absolute and relative absence thresholds
                    if ((absolute_absence > self.absolute_absence_threshold * md.sample_rate) or (relative_absence > self.relative_absence_threshold)):
                        expirations.append(item)
                    elif (item.get_add_active_sample_count(current_annotation) == current_annotation.get_active_sample_count()):
                        match_found = True
                        item.extend(current_annotation)
                        updates.append(item)
                        break
                    
            # add a new signal if no valid matching signals were found
            if not match_found:
                self.active_signals.append(current_annotation)
                updates.append(current_annotation)

            # finally, fill up the inference result annotations with the expirations
            # Python change: have to do the list in reverse order so del works correctly
            for i in range(len(expirations)-1, -1, -1):
                iter = expirations[i]
                if self.expirations:
                    self.out_annotation_count += 1
                    self.final_res_annotations.append(iter.get_annotation(True))

                del expirations[i]

            if self.streaming and hasattr(md, 'is_last_batch_for_token') and not md.is_last_batch_for_token:
                for iter in updates:
                    self.final_res_annotations.append(iter.get_annotation())

        global_expirations = []

        for iter in self.active_signals:
            absolute_absence = iter.get_absolute_absence(self.frame_sequence_index)
            relative_absence = iter.get_relative_absence(self.frame_sequence_index)

            if ((absolute_absence > self.absolute_absence_threshold * md.sample_rate) or (relative_absence > self.relative_absence_threshold)):
                global_expirations.append(iter)

        for iter in global_expirations:
            if self.expirations:
                self.out_annotation_count += 1
                self.final_res_annotations.append(iter.get_annotation(True))

                if len(self.active_signals) > 0:
                    # Iterate backwards so we can cleanly delete
                    for i in range(len(self.active_signals)-1, -1, -1):
                        cur_annotation = self.active_signals[i]
                        if cur_annotation == iter:
                            del self.active_signals[i]
                            break
            
        #if hasattr(md, 'is_last_batch_for_token') and md.is_last_batch_for_token:
        remaining_annotations = self.flush()
        self.out_annotation_count+= len(remaining_annotations)
        self.final_res_annotations += remaining_annotations
        
        # res.annotations = remaining_annotations
        
        self.frame_sequence_index += md.number_samples
        
        return res
        
