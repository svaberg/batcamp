# PyPI Publish Checklist

This checklist is for publishing `batcamp` from `/Users/dagfev/Documents/starwinds/batcamp-main`.

## Metadata and dependencies

- [x] Use the MIT license and include it in package metadata.
- [x] Use the non-deprecated SPDX/license-files metadata format in `pyproject.toml`.
- [x] Add project URLs, classifiers, and keywords to `pyproject.toml`.
- [x] Rename runtime imports from `starwinds_readplt` to `batread`.
- [x] Declare `batread` as a runtime dependency in `pyproject.toml`.
- [x] Decide the minimum supported `batread` version and pin it as `batread>=1.0.0`.
- [x] Use `version = "0.1.0"` as the first PyPI release version.

## Repo hygiene

- [x] Remove the ray feature from the public API, tests, docs, and examples before release.
- [x] Remove generated build artifacts before the release commit.
- [x] Make sure generated artifacts such as `build/` and `dist/` are ignored if they should never be committed.
- [x] Commit the release-ready state on `main`.
- [x] Tag the release commit with the version that will be uploaded.

## Packaging validation

- [x] Make the default pytest run pass against the local `sample_data/` files without needing `pooch`.
- [x] Build both sdist and wheel with `python -m build`.
- [x] Run `twine check dist/*`.
- [x] Test installation from the built wheel in a clean environment that does not see the local `/Users/dagfev/Documents/starwinds/batread` checkout on `sys.path`.
- [x] Verify a clean install of the built wheel pulls `batread` automatically from PyPI.
- [x] Smoke-test `import batcamp` after that clean install.
- [x] Run a small runtime smoke test that exercises `batread.dataset.Dataset` with `batcamp`, not just imports.
- [x] Run the relevant automated tests in a clean environment after installing from built artifacts.

## Docs and install story

- [x] Make the README use the `batread` import path in examples.
- [x] Update `environment.yml` to install `batread` from PyPI instead of a Git URL.
- [x] Read the rendered README once more with PyPI in mind and make sure the install instructions still match the final dependency story.
- [x] Make sure any notebooks or examples you expect users to copy still run with the published dependency names.

## Release execution

- [x] Decide how publishing will happen: manual `twine upload`.
- [x] Make sure the PyPI token or other `twine` credentials are ready before the release tag is pushed.
- [x] Decide to skip TestPyPI for `0.1.0` and upload directly to PyPI.
- [x] Upload the validated tag artifacts to PyPI.
- [x] In a fresh environment, verify `pip install batcamp` from real PyPI works.
