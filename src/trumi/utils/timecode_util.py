import datetime
from fractions import Fraction
from typing import Union

import av

ISO_DATETIME_FMT = r"%Y-%m-%dT%H:%M:%S.%fZ"


def timecode_to_seconds(
    timecode: str, frame_rate: Union[int, float, Fraction]
) -> Union[float, Fraction]:
    """Convert a non-drop-frame timecode string to seconds since midnight.

    Rounds the frame rate to the nearest integer for frame counting (e.g. 29.97 → 30),
    then divides by the true frame rate to preserve sub-frame precision.

    :param timecode: Timecode string in HH:MM:SS:FF format.
    :param frame_rate: True frame rate of the stream.
    :return: Elapsed seconds since midnight as a float or Fraction.
    """
    # calculate whole frame rate
    # 29.97 -> 30, 59.94 -> 60
    int_frame_rate = round(frame_rate)

    # parse timecode string
    h, m, s, f = [int(x) for x in timecode.split(":")]

    # calculate frames assuming whole frame rate (i.e. non-drop frame)
    frames = (3600 * h + 60 * m + s) * int_frame_rate + f

    # convert to seconds
    seconds = frames / frame_rate
    return seconds


def stream_get_start_datetime(stream: av.stream.Stream) -> datetime.datetime:
    """Get the precise start datetime of the first frame in a video stream.

    :param stream: PyAV video stream with timecode and creation_time metadata.
    :return: Timezone-aware datetime of the first frame (UTC).
    """
    # read metadata
    frame_rate = stream.average_rate
    metadata = stream.metadata or {}
    try:
        tc = metadata["timecode"]
        creation_time = metadata["creation_time"]
    except KeyError as exc:
        available_keys = ", ".join(sorted(metadata.keys())) or "<none>"
        raise KeyError(
            f"Required stream metadata key {exc!s} not found. "
            f"Available metadata keys: {available_keys}"
        ) from exc

    # get time within the day
    seconds_since_midnight = float(
        timecode_to_seconds(timecode=tc, frame_rate=frame_rate)
    )
    delta = datetime.timedelta(seconds=seconds_since_midnight)

    # get dates
    create_datetime = datetime.datetime.strptime(creation_time, ISO_DATETIME_FMT)
    create_datetime = create_datetime.replace(
        hour=0, minute=0, second=0, microsecond=0, tzinfo=datetime.timezone.utc
    )
    start_datetime = create_datetime + delta
    return start_datetime


def mp4_get_start_datetime(mp4_path: str) -> datetime.datetime:
    """Return the precise start datetime of the first video frame in an MP4 file.

    :param mp4_path: Path to the MP4 file.
    :return: Timezone-aware datetime of the first frame (UTC).
    """
    with av.open(mp4_path) as container:
        video_streams = container.streams.video
        if not video_streams:
            raise ValueError(f"No video streams found in MP4 file: {mp4_path}")
        stream = video_streams[0]
        return stream_get_start_datetime(stream=stream)
