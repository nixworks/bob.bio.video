from collections import OrderedDict
from bob.bio.face.annotator import Base, min_face_size_validator
from .. import utils
import logging

logger = logging.getLogger(__name__)


class FailSafeVideo(Base):
  """A fail-safe video annotator.
  It tries several annotators in order and tries the next one if the previous
  one fails. However, the difference between this annotator and
  :any:`bob.bio.base.annotator.FailSafe` is that this one tries to use
  annotations from older frames (if valid) before trying the next annotator.

  Attributes
  ----------
  annotators : list
      A list of annotators to try
  max_age : int
      The maximum number of frames that an annotation is valid for next frames.
      This value should be positive. If you want to set max_age to infinite,
      then you can use the :any:`bob.bio.video.annotator.Wrapper` instead.
  required_keys : list
      A list of keys that should be available in annotations to stop trying
      different annotators.
  validator : callable
      A function that takes the annotations and validates it.
  """

  def __init__(self, annotators, required_keys, max_age=20,
               validator=min_face_size_validator, **kwargs):
    super(FailSafeVideo, self).__init__(**kwargs)
    assert max_age > 0, "max_age: `{}' cannot be less than 1".format(max_age)
    self.annotators = list(annotators)
    self.required_keys = list(required_keys)
    self.max_age = max_age
    self.validator = validator

  def annotate(self, frames, **kwargs):
    if isinstance(frames, utils.FrameContainer):
      frames = frames.as_array()
    annotations = OrderedDict()
    current = {}
    age = 0
    for i, frame in enumerate(frames):
      for annotator in self.annotators:
        annot = annotator.annotate(frame, **kwargs)
        if annot and self.validator(annot):
          current = annot
          age = 0
          break
        elif age < self.max_age:
          age += 1
          break
        else:  # no detections and age is larger than maximum allowed
          current = {}

        if current is not annot:
          logger.debug("Annotator `%s' failed.", annotator)

      annotations[str(i)] = current
    return annotations
