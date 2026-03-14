# PyPI Publish Checklist

This checklist is for publishing `batcamp` from `/Users/dagfev/Documents/starwinds/batcamp-main`.

## Metadata and dependencies

- [x] Use the MIT license and include it in package metadata.
- [x] Use the non-deprecated SPDX/license-files metadata format in `pyproject.toml`.
- [x] Add project URLs, classifiers, and keywords to `pyproject.toml`.
- [x] Rename runtime imports from `starwinds_readplt` to `batread`.
- [x] Declare `batread` as a runtime dependency in `pyproject.toml`.
- [x] Decide the minimum supported `batread` version and pin it as `batread>=1.0.0`.
- [ ] Decide whether `version = "0.1.0"` is the actual first PyPI release version or if it should be bumped before release.

## Repo hygiene

- [ ] Remove generated build artifacts before the release commit. There is currently an untracked `build/` directory in this worktree.
- [x] Make sure generated artifacts such as `build/` and `dist/` are ignored if they should never be committed.
- [ ] Commit the release-ready state on `main`.
- [ ] Tag the release commit with the version that will be uploaded.

## Packaging validation

- [x] Build both sdist and wheel with `python -m build`.
- [x] Run `twine check dist/*`.
- [x] Test installation from the built wheel in a clean environment that does not see the local `/Users/dagfev/Documents/starwinds/batread` checkout on `sys.path`.
- [ ] Verify `pip install batcamp` pulls `batread` automatically from PyPI.
- [x] Smoke-test `import batcamp` after that clean install.
- [x] Run a small runtime smoke test that exercises `batread.dataset.Dataset` with `batcamp`, not just imports.
- [x] Run the relevant automated tests in a clean environment after installing from built artifacts.

## Docs and install story

- [x] Make the README use the `batread` import path in examples.
- [x] Update `environment.yml` to install `batread` from PyPI instead of a Git URL.
- [x] Read the rendered README once more with PyPI in mind and make sure the install instructions still match the final dependency story.
- [x] Make sure any notebooks or examples you expect users to copy still run with the published dependency names.

## Release execution

- [ ] Decide how publishing will happen: manual `twine upload` or a trusted-publisher workflow.
- [ ] Make sure the PyPI credentials or trusted-publisher setup are ready before the release tag is pushed.
- [ ] Upload to TestPyPI first.
- [ ] In a fresh environment, verify `pip install -i https://test.pypi.org/simple batcamp` works as expected.
- [ ] Upload the same validated artifacts to PyPI.
- [ ] In one more fresh environment, verify `pip install batcamp` from real PyPI works.
