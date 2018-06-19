# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Produces tfrecord files similar to the YouTube-8M dataset.

It processes a CSV file containing lines like "<video_file>,<labels>", where
<video_file> must be a path of a video, and <labels> must be an integer list
joined with semi-colon ";". It processes all videos and outputs tfrecord file
onto --output_tfrecords_file.

It assumes that you have OpenCV installed and properly linked with ffmpeg (i.e.
function `cv2.VideoCapture().open('/path/to/some/video')` should return True).

The binary only processes the video stream (images) and not the audio stream.
"""

import os
import sys

import cv2
import feature_extractor
import numpy
import tensorflow as tf
import youtube_dl

# In OpenCV3.X, this is available as cv2.CAP_PROP_POS_MSEC
# In OpenCV2.X, this is available as cv2.cv.CV_CAP_PROP_POS_MSEC
CAP_PROP_POS_MSEC = 0

def extract_features(output_tfrecords_file,
    videos_urls,
    videos_labels,
    streaming = False,
    model_dir = 'yt8m',
    frames_per_second = 1,
    skip_frame_level_features = False,
    labels_feature_key = 'labels',
    image_feature_key = 'rgb',
    video_file_feature_key = 'id',
    insert_zero_audio_features = True):

    extractor = feature_extractor.YouTube8MFeatureExtractor(model_dir)
    writer = tf.python_io.TFRecordWriter(output_tfrecords_file)
    total_written = 0
    total_error = 0
    ydl = youtube_dl.YoutubeDL({})
    for video, labels in zip(videos_urls, videos_labels):
        rgb_features = []
        sum_rgb_features = None
        for rgb in frame_iterator(video, ydl, streaming, every_ms=1000.0 / frames_per_second):
            features = extractor.extract_rgb_frame_features(rgb[:, :, ::-1])
            if sum_rgb_features is None:
                sum_rgb_features = features
            else:
                sum_rgb_features += features
            rgb_features.append(_bytes_feature(quantize(features)))

        if not rgb_features:
            print >> sys.stderr, 'Could not get features for ' + video
            total_error += 1
            continue

        mean_rgb_features = sum_rgb_features / len(rgb_features)

        # Create SequenceExample proto and write to output.
        feature_list = {
            image_feature_key: tf.train.FeatureList(feature=rgb_features),
        }
        context_features = {
            labels_feature_key: _int64_list_feature(
                sorted(map(int, labels.split(',')))),
            video_file_feature_key: _bytes_feature(_make_bytes(
                list(map(ord, video)))),
            'mean_' + image_feature_key: tf.train.Feature(
                float_list=tf.train.FloatList(value=mean_rgb_features)),
        }

        if insert_zero_audio_features:
            zero_vec = [0] * 128
            feature_list['audio'] = tf.train.FeatureList(
                feature=[_bytes_feature(_make_bytes(zero_vec))] * len(rgb_features))
            context_features['mean_audio'] = tf.train.Feature(
                float_list=tf.train.FloatList(value=zero_vec))

        if skip_frame_level_features:
            example = tf.train.SequenceExample(
                context=tf.train.Features(feature=context_features))
        else:
            example = tf.train.SequenceExample(
                context=tf.train.Features(feature=context_features),
                feature_lists=tf.train.FeatureLists(feature_list=feature_list))
        writer.write(example.SerializeToString())
        total_written += 1

    writer.close()
    print(('Successfully encoded %i out of %i videos' % (
        total_written, total_written + total_error)))

def frame_iterator(video_url, ydl, streaming, every_ms=1000, max_num_frames=300):
    """Uses OpenCV to iterate over all frames of filename at a given frequency.

    Args:
      filename: Path to video file (e.g. mp4)
      every_ms: The duration (in milliseconds) to skip between frames.
      max_num_frames: Maximum number of frames to process, taken from the
        beginning of the video.

    Yields:
      RGB frame with shape (image height, image width, channels)
    """
    if streaming:
        # for loading an online video directly into the memory
        info_dict = ydl.extract_info(video_url, download=False)
        format = info_dict.get('formats', None)[-1]
        url = format.get('url', None)
        video_capture = cv2.VideoCapture(url)
    else:
        video_capture = cv2.VideoCapture(video_url)
    if not video_capture.isOpened():
        print >> sys.stderr, 'Error: Cannot open video url ' + video_url
        return
    last_ts = -99999  # The timestamp of last retrieved frame.
    num_retrieved = 0

    while num_retrieved < max_num_frames:
        # Skip frames
        while video_capture.get(CAP_PROP_POS_MSEC) < every_ms + last_ts:
            if not video_capture.read()[0]:
                return

        last_ts = video_capture.get(CAP_PROP_POS_MSEC)
        has_frames, frame = video_capture.read()
        if not has_frames:
            break
        yield frame
        num_retrieved += 1


def _int64_list_feature(int64_list):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=int64_list))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _make_bytes(int_array):
    if bytes == str:  # Python2
        return ''.join(map(chr, int_array))
    else:
        return bytes(int_array)


def quantize(features, min_quantized_value=-2.0, max_quantized_value=2.0):
    """Quantizes float32 `features` into string."""
    assert features.dtype == 'float32'
    assert len(features.shape) == 1  # 1-D array
    features = numpy.clip(features, min_quantized_value, max_quantized_value)
    quantize_range = max_quantized_value - min_quantized_value
    features = (features - min_quantized_value) * (255.0 / quantize_range)
    features = [int(round(f)) for f in features]

    return _make_bytes(features)
