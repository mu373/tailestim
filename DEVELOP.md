# Development notes

## Building package
```sh
hatch run test:all
hatch build
```

## GitHub Actions
Tests and releases are automated using GitHub Actions.
- `test.yml`: Test CI
   - Runs when new pushes and pull requests are made in `main` branch.
   - Runs tests in different Python versions
- `release.yml`: Release CI
   - Use [this GitHub Actions](https://github.com/mu373/tailestim/actions/workflows/release.yml) to make a new release. Manually click on the "Run Workflow" button and enter the new version name.
   - Updates version in `__about__.py` and commits to the `main` branch.
   - Tags the commit and makes a [release](https://github.com/mu373/tailestim/releases) in GitHub
   - Builds and publishes to [PyPI](https://pypi.org/project/powerlaw/)

## conda-forge
- Once a release is pubslihed to [PyPI](https://pypi.org/project/tailestim/), a bot (`@regro-cf-autotick-bot`) will automatically detect the new version and publishes it to [conda-forge](https://anaconda.org/conda-forge/tailestim). This could take some time (few hours).
- Recipe is available at [conda-forge/tailestim-feedstock](https://github.com/conda-forge/tailestim-feedstock). 
