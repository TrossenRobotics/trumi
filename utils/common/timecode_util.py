import datetime
from fractions import Fraction
from typing import Union

import av


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
    :return: Datetime of the first frame in UTC.
    """
    # read metadata
    frame_rate = stream.average_rate
    tc = stream.metadata["timecode"]
    creation_time = stream.metadata["creation_time"]

    # get time within the day
    seconds_since_midnight = float(
        timecode_to_seconds(timecode=tc, frame_rate=frame_rate)
    )
    delta = datetime.timedelta(seconds=seconds_since_midnight)

    # get dates
    create_datetime = datetime.datetime.strptime(
        creation_time, r"%Y-%m-%dT%H:%M:%S.%fZ"
    )
    create_datetime = create_datetime.replace(hour=0, minute=0, second=0, microsecond=0)
    start_datetime = create_datetime + delta
    return start_datetime


def mp4_get_start_datetime(mp4_path: str) -> datetime.datetime:
    """Return the precise start datetime of the first video frame in an MP4 file.

    :param mp4_path: Path to the MP4 file.
    :return: Datetime of the first frame in UTC.
    """
    with av.open(mp4_path) as container:
        stream = container.streams.video[0]
        return stream_get_start_datetime(stream=stream)
