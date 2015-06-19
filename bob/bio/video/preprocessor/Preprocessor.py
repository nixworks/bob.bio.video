import bob.bio.base
import numpy
import glob
import os

import bob.io.base

from .. import utils

class Preprocessor (bob.bio.base.preprocessor.Preprocessor):

  def __init__(self,
      preprocessor = 'landmark-detect',
      frame_selector = utils.FrameSelector(),
      quality_function = None,
      compressed_io = False
  ):

    # load preprocessor configuration
    if isinstance(preprocessor, str):
      self.preprocessor = bob.bio.base.load_resource(preprocessor, "preprocessor")
    elif isinstance(preprocessor, bob.bio.base.preprocessor.Preprocessor):
      self.preprocessor = preprocessor
    else:
      raise ValueError("The given algorithm could not be interpreter")

    bob.bio.base.preprocessor.Preprocessor.__init__(
        self,
        preprocessor=preprocessor,
        frame_selector=frame_selector,
        compressed_io=compressed_io
    )

    self.frame_selector = frame_selector
    self.quality_function = quality_function
    self.compressed_io = compressed_io

  def _check_feature(self, frames):
    assert isinstance(frames, utils.FrameContainer)


  def __call__(self, frames, annotations=None):
    """Extracts the frames from the video and returns a frame container.

    Faces are extracted for all frames in the given frame container, using the ``preprocessor`` specified in the contructor.

    If given, the annotations need to be in a dictionary.
    The key is either the frame number (for video data) or the image name (for image list data).
    The value is another dictionary, building the relation between keypoint names and their location, e.g., {'leye' : (le_y, le_x), 'reye' : (re_y, re_x)}
    The annotations for the according frames, if present, are passed to the preprocessor.
    """

    self._check_feature(frames)

    annots = None
    fc = utils.FrameContainer()

    for index, frame, _ in frames:
      # if annotations are given, we take them
      if annotations is not None: annots = annotations[index]

      # preprocess image (by default: detect a face)
      preprocessed = self.preprocessor(frame, annots)
      if preprocessed is not None:
        # compute the quality of the detection
        if self.quality_function is not None:
          quality = self.quality_function(preprocessed)
        elif hasattr(self.preprocessor, 'quality'):
          quality = self.preprocessor.quality
        else:
          quality = None
        # add image to frame container
        if hasattr(preprocessed, 'copy'):
          preprocessed = preprocessed.copy()
        fc.add(index, preprocessed, quality)

    return fc


  def read_original_data(self, data):
    return self.frame_selector(data)

  def read_data(self, filename):
    if self.compressed_io:
      return utils.load_compressed(filename, self.preprocessor.read_data)
    else:
      return utils.FrameContainer(bob.io.base.HDF5File(filename), self.preprocessor.read_data)

  def write_data(self, frames, filename):
    self._check_feature(frames)

    if self.compressed_io:
      return utils.save_compressed(frames, filename, self.preprocessor.write_data)
    else:
      frames.save(bob.io.base.HDF5File(filename, 'w'), self.preprocessor.write_data)
