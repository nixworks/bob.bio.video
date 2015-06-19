import bob.bio.base
import bob.io.base

from .. import utils

class Algorithm (bob.bio.base.algorithm.Algorithm):

  def __init__(self,
      algorithm,
      frame_selector = utils.FrameSelector(selection_style='all'),
      compressed_io = False
  ):
    # load algorithm configuration
    if isinstance(algorithm, str):
      self.algorithm = bob.bio.base.load_resource(algorithm, "algorithm")
    elif isinstance(algorithm, bob.bio.base.algorithm.Algorithm):
      self.algorithm = algorithm
    else:
      raise ValueError("The given algorithm could not be interpreter")

    bob.bio.base.algorithm.Algorithm.__init__(
        self,
        self.algorithm.performs_projection,
        self.algorithm.requires_projector_training,
        self.algorithm.split_training_features_by_client,
        self.algorithm.use_projected_features_for_enrollment,
        self.algorithm.requires_enroller_training,
        algorithm=algorithm,
        frame_selector=frame_selector,
        compressed_io=compressed_io
    )

    self.frame_selector = frame_selector
    # if we select the frames during feature extraction, for enrollment we use all files
    # otherwise select frames during enrollment (or enroller training)
    self.enroll_frame_selector = (lambda i : i) if self.use_projected_features_for_enrollment else frame_selector
    self.compressed_io = compressed_io


  def _check_feature(self, frames):
    assert isinstance(frames, utils.FrameContainer)

  # PROJECTION

  def train_projector(self, data_list, projector_file):
    """Trains the projector using features from selected frames."""
    if self.split_training_features_by_client:
      [self._check_feature(frames) for client_frames in data_list for frames in client_frames]
      training_features = [[frame[1] for frames in client_frames for frame in self.frame_selector(frames)] for client_frames in data_list]
    else:
      [self._check_feature(frames) for frames in data_list]
      training_features = [frame[1] for frames in data_list for frame in self.frame_selector(frames)]
    self.algorithm.train_projector(training_features, projector_file)


  def load_projector(self, projector_file):
    return self.algorithm.load_projector(projector_file)


  def project(self, frames):
    """Projects each frame and saves them in a frame container."""
    self._check_feature(frames)
    fc = utils.FrameContainer()
    for index, frame, quality in self.frame_selector(frames):
      # extract features
      projected = self.algorithm.project(frame)
      features = projected if isinstance(projected, (list,tuple)) else projected.copy()
      # add image to frame container
      fc.add(index, features, quality)
    return fc


  def write_feature(self, frames, projected_file):
    self._check_feature(frames)
    if self.compressed_io:
      return utils.save_compressed(frames, projected_file, self.algorithm.write_feature)
    else:
      frames.save(bob.io.base.HDF5File(projected_file, 'w'), self.algorithm.write_feature)

  def read_feature(self, projected_file):
    if self.compressed_io:
      return utils.load_compressed(projected_file, self.algorithm.read_feature)
    else:
      return utils.FrameContainer(bob.io.base.HDF5File(projected_file), self.algorithm.read_feature)


  # ENROLLMENT

  def train_enroller(self, training_frames, enroller_file):
    [self._check_feature(frames) for client_frames in training_frames for frames in client_frames]
    features = [[frame[1] for frames in client_frames for frame in self.enroll_frame_selector(frames)] for client_frames in training_frames]
    self.algorithm.train_enroller(features, enroller_file)

  def load_enroller(self, enroller_file):
    self.algorithm.load_enroller(enroller_file)


  def enroll(self, enroll_frames):
    """Enrolls the model from features of all images of all videos."""
    [self._check_feature(frames) for frames in enroll_frames]
    features = [frame[1] for frames in enroll_frames for frame in self.enroll_frame_selector(frames)]
    return self.algorithm.enroll(features)

  def write_model(self, model, filename):
    """Saves the model using the algorithm's save function."""
    self.algorithm.write_model(model, filename)


  # SCORING

  def read_model(self, filename):
    """Reads the model using the algorithm's read function."""
    return self.algorithm.read_model(filename)

  def read_probe(self, filename):
    """Reads the model using the algorithm's read function."""
    # TODO: check if it is really necessary that we read other types than FrameContainers here...
    try:
      if self.compressed_io:
        return utils.load_compressed(filename, self.algorithm.read_probe)
      else:
        return utils.FrameContainer(bob.io.base.HDF5File(filename), self.algorithm.read_probe)
    except IOError:
      return self.algorithm.read_probe(filename)

  def score(self, model, probe):
    """Computes the score between the given model and the probe, which is a list of frames."""
    # TODO: check if it is really necessary that we treat other types than FrameContainers here...
    if isinstance(probe, utils.FrameContainer):
      features = [frame[1] for frame in probe]
      return self.algorithm.score_for_multiple_probes(model, features)
    else:
      return self.algorithm.score(model, probe)


  def score_for_multiple_probes(self, model, probes):
    """Computes the score between the given model and the probes, where each probe is a list of frames."""
    [self._check_feature(frames) for frames in probes]
    probe = [frame[1] for frame in probe for probe in probes]
    return self.algorithm.score_for_multiple_probes(model, probe)
