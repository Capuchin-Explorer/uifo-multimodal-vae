# GitHub release checklist

- [ ] Confirm that all absolute HPC paths were removed or made configurable.
- [ ] Exclude `.idea/`, `.pyc`, logs, checkpoints, and private blacklist files.
- [ ] Add the three representation matrices, vocabularies, indexes, and shared metadata.
- [ ] Track large `.npy` and `.parquet` files with Git LFS or attach them to a release.
- [ ] Add SHA-256 checksums for released data artifacts.
- [ ] Run `python -m py_compile src/*.py scripts/build_representations.py`.
- [ ] Test one short training run and one visualization run.
- [ ] Add the final thesis citation and repository URL before generating the presentation QR code.
