import six
from collections import OrderedDict
from bob.bio.face.annotator import Base
from bob.bio.base import load_resource
from bob.bio.face.annotator import min_face_size_validator

from .. import utils
from . import normalize_annotations


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
    if isinstance(annotator, six.string_types):
      self.annotator = load_resource(annotator, "annotator")

  def annotate(self, frames, **kwargs):
    if isinstance(frames, utils.FrameContainer):
      frames = frames.as_array()
    annotations = OrderedDict()
    for i, frame in enumerate(frames):
      annotations[str(i)] = self.annotator(frame, **kwargs)
    if self.normalize:
      annotations = OrderedDict(normalize_annotations(
          annotations, self.validator, self.max_age))
    return annotations
