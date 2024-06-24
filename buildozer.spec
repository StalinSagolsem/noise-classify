[app]
title = Audio Classifier
package.name = audioclassifier
package.domain = org.test
source.dir = .
source.include_exts = py,png,jpg,kv,atlas,h5
requirements = python3,kivy,numpy,librosa,tensorflow,audiostream
version = 0.1

android.permissions = WRITE_EXTERNAL_STORAGE, READ_EXTERNAL_STORAGE

[buildozer]
log_level = 2
warn_on_root = 1