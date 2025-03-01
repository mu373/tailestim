# Development notes

## Building package
```sh
hatch run test:all
hatch build
```

## GitHub Actions
- `test.yml`: Test functions
- `release.yml`: Automatically publish to PyPI when a release is made
