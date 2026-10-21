"""Beat detector vendored from pyvital 0.4.0 (``pyvital.arr.detect_peaks``).

pyvital 0.5.0 removed the ``arr`` submodule, and 0.4.0 pulls in tensorflow and
torch as hard requirements. The detector and its helpers are copied here
verbatim so results match pyvital 0.4.0 exactly, with two non-numerical edits:

* the "HR estimation failed" ``print`` now goes through :mod:`logging`;
* unused helpers and commented-out code were dropped.

Original source: https://github.com/vitaldb/pyvital (MIT License)

    Copyright (c) 2021 VitalDB

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""
import logging
import math

import numpy as np
from scipy.signal import argrelextrema, butter, filtfilt

logger = logging.getLogger(__name__)


def max_idx(data, idxfrom=0, idxto=None):
    idxfrom = max(0, idxfrom)
    if idxto is None:
        idxto = len(data)
    idxto = min(len(data), idxto)
    return idxfrom + np.argmax(data[idxfrom:idxto])


def min_idx(data, idxfrom=0, idxto=None):
    idxfrom = max(0, idxfrom)
    if idxto is None:
        idxto = len(data)
    idxto = min(len(data), idxto)
    return idxfrom + np.argmin(data[idxfrom:idxto])


def resample_hz(data, srate_from, srate_to, avg=False):
    dest_len = int(math.ceil(len(data) / srate_from * srate_to))
    return resample(data, dest_len, avg)


def resample(data, dest_len, avg=False):
    """Resample to ``dest_len`` points (linear up-sampling, nearest/mean down-sampling)."""
    if dest_len == 0:
        return []

    src_len = len(data)
    if src_len == 0:
        return np.zeros(dest_len)

    if dest_len == 1:  # average
        if avg:
            return np.array([np.mean(data)])
        else:
            return np.array([data[0]])

    if src_len == 1:  # copy
        return np.full(dest_len, data[0])

    if not isinstance(data, np.ndarray):
        data = np.array(data)

    if src_len == dest_len:
        return np.copy(data)

    if src_len < dest_len:  # upsample -> linear interpolate
        ret = []
        for x in range(dest_len):
            srcx = x / (dest_len - 1) * (src_len - 1)
            srcx1 = math.floor(srcx)
            srcx2 = math.ceil(srcx)
            factor = srcx - srcx1
            val1 = data[srcx1]
            val2 = data[srcx2]
            ret.append(val1 * (1 - factor) + val2 * factor)
        return np.array(ret)

    if avg:  # downsample -> mean
        ret = []
        for x in range(dest_len):
            src_from = int(x * src_len / dest_len)
            src_to = int((x + 1) * src_len / dest_len)
            ret.append(np.mean(data[src_from:src_to]))
        return np.array(ret)

    ret = []  # downsample -> nearest
    for x in range(dest_len):
        srcx = int(x * src_len / dest_len)
        ret.append(data[srcx])
    return np.array(ret)


def band_pass(data, srate, fl, fh, order=5):
    if fl > fh:
        return band_pass(data, srate, fh, fl)
    nyq = 0.5 * srate
    b, a = butter(order, [fl / nyq, fh / nyq], btype='band')
    return filtfilt(b, a, data)


def estimate_heart_freq(data, srate, fl=30 / 60, fh=200 / 60):
    """Harmonic-sum PSD heart-rate estimate (Aboy et al., IEEE TBME 2005; PMID 16235652)."""
    p = np.abs(np.fft.fft(data)) ** 2
    maxf = 0
    maxval = 0
    for w in range(len(data)):
        f = w * srate / len(data)
        # add 11 harmonics, which do not exceed double of the default power sampling
        if fl <= f <= fh:
            h = 0
            for k in range(1, 12):
                h += min(2 * p[w], p[(k * w) % len(data)])
            if h > maxval:
                maxval = h
                maxf = f
    return maxf


def detect_maxima(data, tr=None):
    """Indices with ``x[k-1] <= x[k] >= x[k+1]``, optionally above the ``tr``-th percentile."""
    if tr is None:
        return np.array(argrelextrema(data, np.greater_equal)[0])
    tval = np.percentile(data, tr)
    ret = []
    for i in argrelextrema(data, np.greater_equal)[0]:
        if data[i] > tval:
            ret.append(i)
    return np.array(ret)


def find_nearest(a, value):
    """Nearest value in a *sorted* array."""
    idx = np.searchsorted(a, value)
    if idx > 0 and (idx == len(a) or math.fabs(value - a[idx - 1]) < math.fabs(value - a[idx])):
        return a[idx - 1]
    else:
        return a[idx]


def detect_peaks(data, srate):
    """Systolic peaks and diastolic feet of a pressure/pleth waveform.

    Returns ``[minlist, maxlist]``; ``len(minlist) == len(maxlist) - 1`` and each
    foot is the minimum between two neighbouring peaks.
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    raw_data = np.copy(data)
    raw_srate = srate

    # resampling rate to 100Hz
    data = resample_hz(data, srate, 100)
    srate = 100

    # upper and lower bound of the heart rate (Hz = /sec)
    fh = 200 / 60  # 3.3
    fl = 30 / 60  # 0.5

    # estimate hr
    y1 = band_pass(data, srate, 0.5 * fl, 3 * fh)

    # Whole heart freq estimation
    hf = estimate_heart_freq(y1, srate)
    if hf == 0:
        logger.debug("HR estimation failed, assume 75")
        hf = 75 / 60

    # band pass filter again with heart freq estimation
    y2 = band_pass(data, srate, 0.5 * fl, 2.5 * hf)
    d2 = np.diff(y2)

    # detect peak in gradient
    p2 = detect_maxima(d2, 90)

    # detect real peak
    y3 = band_pass(data, srate, 0.5 * fl, 10 * hf)
    p3 = detect_maxima(y3, 60)

    # find closest p3 that follows p2
    p4 = []
    last_p3 = 0
    for idx_p2 in p2:
        idx_p3 = 0
        for idx_p3 in p3:
            if idx_p3 > idx_p2:
                break
        if idx_p3 != 0:
            if last_p3 != idx_p3:
                p4.append(idx_p3)
                last_p3 = idx_p3

    # nearest neighbor and inter beat interval correction
    pc = []

    # find all maxima before preprocessing
    m = detect_maxima(data, 0)
    m = np.array(m)

    # correct peaks location error due to preprocessing
    last = -1
    for idx_p4 in p4:
        cand = find_nearest(m, idx_p4)
        if cand != last:
            pc.append(cand)
            last = cand

    ht = 1 / hf  # beat interval (sec)

    # correct false negatives (FN)
    i = -1
    while i < len(pc):
        if i < 0:
            idx_from = 0
        else:
            idx_from = pc[i]

        if i >= len(pc) - 1:
            idx_to = len(data) - 1
        else:
            idx_to = pc[i + 1]

        # find false negative and fill it
        if idx_to - idx_from < 1.75 * ht * srate:
            i += 1
            continue

        # It can not be within 0.2 of both sides
        idx_from += 0.2 * ht * srate
        idx_to -= 0.2 * ht * srate

        # find the maximum value from idx_from to idx_to
        idx_max = -1
        val_max = 0

        for j in range(np.searchsorted(m, idx_from), len(m)):
            idx_cand = m[j]
            if idx_cand >= idx_to:
                break
            if idx_max == -1 or val_max < data[idx_cand]:
                val_max = data[idx_cand]
                idx_max = idx_cand

        if idx_max != -1:  # add idx_max and restart from there
            pc.insert(i + 1, idx_max)
            i -= 1
        i += 1

    # correct false positives (FP)
    i = 0
    while i < len(pc) - 1:
        idx1 = pc[i]
        idx2 = pc[i + 1]
        if idx2 - idx1 < 0.75 * ht * srate:  # false positive
            idx_del = i + 1  # default: delete i+1
            if 1 < i < len(pc) - 2:
                # minimize heart rate variability
                idx_prev = pc[i - 1]
                idx_next = pc[i + 2]

                d1 = abs(idx_next + idx_prev - 2 * idx1)
                d2 = abs(idx_next + idx_prev - 2 * idx2)

                if d1 > d2:
                    idx_del = i
                else:
                    idx_del = i + 1

            elif i == 0:
                idx_del = i
            elif i == len(pc) - 2:
                idx_del = i + 1

            pc.pop(idx_del)
            i -= 1
        i += 1

    # remove duplicates
    i = 0
    for i in range(0, len(pc) - 1):
        if pc[i] == pc[i + 1]:
            pc.pop(i)
            i -= 1
        i += 1

    # find nearest peak in real data (positions found at 100 Hz may be off by raw_srate / srate)
    maxlist = []
    ratio = math.ceil(raw_srate / srate)
    for maxidx in pc:
        idx = int(maxidx * raw_srate / srate)
        maxlist.append(max_idx(raw_data, idx - ratio - 1, idx + ratio + 1))

    # get the minlist from maxlist
    minlist = []
    for i in range(len(maxlist) - 1):
        minlist.append(min_idx(raw_data, maxlist[i], maxlist[i + 1]))

    return [minlist, maxlist]
