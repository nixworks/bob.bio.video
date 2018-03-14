import bob.bio.base
from .. import utils


class Base(bob.bio.base.annotator.Annotator):
  """The base class for video annotators.

  Parameters
  ----------
  frame_selector : :any:`bob.bio.video.FrameSelector`
    A frame selector class to define, which frames of the video to use.
  read_original_data : callable
    A function with the signature of
    ``data = read_original_data(biofile, directory, extension)``
    that will be used to load the data from biofiles. By default the
    ``frame_selector`` is used to load the data.
  """

  def __init__(self, frame_selector=utils.FrameSelector(selection_style='all'),
               read_original_data=None, **kwargs):

    def _read_video_data(biofile, directory, extension):
      """Read video data using the frame_selector of this object"""
      return biofile.load(directory, extension, frame_selector)

    if read_original_data is None:
      read_original_data = _read_video_data

    super(Base, self).__init__(read_original_data=read_original_data, **kwargs)

  @staticmethod
  def frame_and_frame_ids(frames):
    """Normalizes the frames parameter into a numpy array and a list of
    indices.

    Parameters
    ----------
    frames : :any:`bob.bio.video.FrameContainer` or :any:`numpy.array`
        The frames of the video file.

    Returns
    -------
    frames : :any:`numpy.array`
        The frames of the video file as a numpy array.
    frame_ids : list
        A list of strings that represent the frame ids.
    """
    if isinstance(frames, utils.FrameContainer):
      frame_ids = [fid for fid, _, _ in frames]
      frames = frames.as_array()
    else:  # frames is already a numpy.array
      frame_ids = [str(i) for i in range(len(frames))]

    return frames, frame_ids

  def annotate(self, frames, **kwargs):
    """Annotates videos.

    Parameters
    ----------
    frames : :any:`bob.bio.video.FrameContainer` or :any:`numpy.array`
        The frames of the video file.
    **kwargs
        Extra arguments that annotators may need.

    Returns
    -------
    collections.OrderedDict
        A dictionary where its key is the frame id as a string and its value
        is a dictionary that are the annotations for that frame.


    .. note::

        You can use the :any:`Base.frame_and_frame_ids` functions to normalize
        the input in your implementation.
    """
    raise NotImplementedError()
