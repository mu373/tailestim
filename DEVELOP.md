# Development notes

## Building package
```sh
hatch run test:all
hatch build
```

## GitHub Actions
- `test.yml`: Test functions
- `release.yml`: Automatically publish to PyPI when a [release](https://github.com/mu373/tailestim/releases) is published. This will run:
   - Update version in `__about__.py`
   - Run tests
   - Build and publish to PyPI
   - Commit updated `__about__.py`

