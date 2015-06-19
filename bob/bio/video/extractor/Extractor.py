import bob.bio.base
import bob.io.base
import os

from .. import utils

class Extractor (bob.bio.base.extractor.Extractor):

  def __init__(self,
      extractor,
      frame_selector = utils.FrameSelector(selection_style='all'),
      compressed_io = False
  ):
    # load extractor configuration
    if isinstance(extractor, str):
      self.extractor = bob.bio.base.load_resource(extractor, "extractor")
    elif isinstance(extractor, bob.bio.base.extractor.Extractor):
      self.extractor = extractor
    else:
      raise ValueError("The given extractor could not be interpreter")

    self.frame_selector = frame_selector
    self.compressed_io = compressed_io
    # register extractor's details
    bob.bio.base.extractor.Extractor.__init__(
        self,
        requires_training=self.extractor.requires_training,
        split_training_data_by_client=self.extractor.split_training_data_by_client,
        extractor=extractor,
        frame_selector=frame_selector,
        compressed_io=compressed_io
    )

  def _check_feature(self, frames):
    assert isinstance(frames, utils.FrameContainer)


  def __call__(self, frames, annotations=None):
    """Extracts the frames from the video and returns a frame container."""
    self._check_feature(frames)
    # go through the frames and extract the features
    fc = utils.FrameContainer()
    for index, frame, quality in self.frame_selector(frames):
      # extract features
      extracted = self.extractor(frame)
      # add features to new frame container
      fc.add(index, extracted, quality)
    return fc


  def read_feature(self, filename):
    if self.compressed_io:
      return utils.load_compressed(filename, self.extractor.read_feature)
    else:
      return utils.FrameContainer(bob.io.base.HDF5File(filename), self.extractor.read_feature)

  def write_feature(self, frames, filename):
    self._check_feature(frames)
    if self.compressed_io:
      return utils.save_compressed(frames, filename, self.extractor.write_feature)
    else:
      frames.save(bob.io.base.HDF5File(filename, 'w'), self.extractor.write_feature)


  def train(self, data_list, extractor_file):
    """Trains the feature extractor with the image data of the given frames."""
    if self.split_training_data_by_client:
      [self._check_feature(frames) for client_frames in data_list for frames in client_frames]
      features = [[frame[1] for frames in client_frames for frame in self.frame_selector(frames)] for client_frames in data_list]
    else:
      [self._check_feature(frames) for frames in data_list]
      features = [frame[1] for frames in data_list for frame in self.frame_selector(frames)]
    self.extractor.train(features, extractor_file)

  def load(self, extractor_file):
    self.extractor.load(extractor_file)
