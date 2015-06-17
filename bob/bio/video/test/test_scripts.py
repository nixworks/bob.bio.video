from __future__ import print_function

import bob.bio.base
import tempfile

from bob.bio.base.test.test_scripts import _verify

def test_verify_video():
  test_dir = tempfile.mkdtemp(prefix='bobtest_')
  # define dummy parameters
  parameters = [
      '-d', 'dummy-video',
      '-p', 'bob.bio.video.preprocessor.Preprocessor("dummy")',
      '-e', 'bob.bio.video.extractor.Extractor("dummy")',
      '-a', 'bob.bio.video.algorithm.Algorithm("dummy")',
      '--zt-norm',
      '-s', 'test_video',
      '--temp-directory', test_dir,
      '--result-directory', test_dir,
      '--imports', 'bob.bio.video'
  ]

  print (bob.bio.base.tools.command_line(parameters))

  _verify(parameters, test_dir, 'test_video')
