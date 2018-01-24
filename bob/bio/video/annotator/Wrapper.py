from bob.bio.face.annotator import Base
from bob.ip.facedetect import bounding_box_from_annotation
from bob.bio.base import load_resource

from .. import utils


def normalize_annotations(annotations, validator, max_age=-1):
  """Normalizes the annotations of one video sequence. It fills the
  annotations for frames from previous ones if the annotation for the current
  frame is not valid.

  Parameters
  ----------
  annotations : dict
      A dict of dict where the keys to the first dict are frame indices as
      strings (starting from 0). The inside dicts contain annotations for
      that frame.
  validator : callable
      Takes a dict (annotations) and returns True if the annotations are
      valid. This can be check based on minimal face size for example.
  max_age : :obj:`int`, optional
      An integer indicating for a how many frames a detected face is valid if
      no detection occurs after such frame. A value of -1 == forever

  Yields
  ------
  str
      The index of frame.
  dict
      The corrected annotations of the frame.
  """
  # the annotations for the current frame
  current = {}
  age = 0

  for k, annot in annotations.items():
    if validator(annot):
      current = annot
      age = 0
    elif max_age < 0 or age < max_age:
      age += 1
    else:  # no detections and age is larger than maximum allowed
      current = {}

    yield k, current


def min_face_size_validator(annotations, min_face_size=32):
  """Validates annotations based on face's minimal size.

  Parameters
  ----------
  annotations : dict
      The annotations in dictionary format.
  min_face_size : int, optional
      The minimal size of a face.

  Returns
  -------
  bool
      True, if the face is large enough.
  """
  bbx = bounding_box_from_annotation(**annotations)
  if bbx.size < 32:
    return False
  return True


class Wrapper(Base):
  """Annotates video files.
  This annotator does not support annotating only select frames of a video.

  Attributes
  ----------
  annotator : :any:`Base`
      The image annotator to be used.
  max_age : int
      see :any:`normalize_annotations`.
  normalize : bool
      If True, it will normalize annotations using :any:`normalize_annotations`
  validator : object
      see :any:`normalize_annotations` and :any:`min_face_size_validator`.
  """

  def __init__(self,
               annotator,
               normalize=True,
               validator=min_face_size_validator,
               max_age=-1,
               **kwargs
               ):
    super(Wrapper, self).__init__(**kwargs)
    self.annotator = annotator
    self.normalize = normalize
    self.validator = validator
    self.max_age = max_age

    # load annotator configuration
    if isinstance(annotator, basestring):
      self.annotator = load_resource(annotator, "annotator")

  def annotate(self, frames, **kwargs):
    if isinstance(frames, utils.FrameContainer):
      frames = frames.as_array()
    annotations = {}
    for i, frame in enumerate(frames):
      annotations[str(i)] = self.annotator(frame, **kwargs)
    if self.normalize:
      annotations = dict(normalize_annotations(
          annotations, self.validator, self.max_age))
    return annotations
