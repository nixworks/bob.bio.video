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
    bob.bio.base.extractor.Extractor.__init__(self, requires_training=self.extractor.requires_training, split_training_data_by_client=self.extractor.split_training_data_by_client, extractor=extractor, frame_selector=frame_selector, compressed_io=compressed_io)


  def __call__(self, frame_container, annotations=None):
    """Extracts the frames from the video and returns a frame container."""
    # now, go through the frames and extract the features
    fc = utils.FrameContainer()
    for index, frame, quality in self.frame_selector(frame_container):
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

  def write_feature(self, frame_container, filename):
    if self.compressed_io:
      return utils.save_compressed(frame_container, filename, self.extractor.write_feature)
    else:
      frame_container.save(bob.io.base.HDF5File(filename, 'w'), self.extractor.save_feature)


  def train(self, data_list, extractor_file):
    """Trains the feature extractor with the image data of the given frames."""
    if self.split_training_data_by_client:
      features = [[frame[1] for frame_container in client_containers for frame in self.frame_selector(frame_container)] for client_containers in data_list]
    else:
      features = [frame[1] for frame_container in data_list for frame in self.frame_selector(frame_container)]
    self.extractor.train(features, extractor_file)

  def load(self, extractor_file):
    self.extractor.load(extractor_file)
