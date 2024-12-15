echo '# Working with Conflicting Python Package Dependencies in Poetry

## The Problem
When working with both a local fork of a package (`docling-core`) and its parent package (`docling`), Poetry'\''s dependency resolver will block the installation due to version conflicts. This is because:
- `docling` requires a specific version of `docling-core`
- Your local fork of `docling-core` is treated as a different version
- Poetry'\''s default behavior is to prevent such conflicts

## Solutions Attempted

### 1. Local Path in pyproject.toml (Didn'\''t Work)
toml
[tool.poetry.dependencies]
docling-core = {path = "docling-core"}
docling = "^2.8.3"


This failed because Poetry detected the version conflict.

### 2. Modifying Package Structure (Not Ideal)
Attempting to restructure the local fork to match expected package structure would require significant changes to the local fork.

### 3. Working Solution: Poetry Shell
The best solution found was to:
1. Enter Poetry'\''s virtual environment shell:

bash
poetry shell

2. Install the conflicting package directly:

bash
pip install docling


This approach works because:
- It maintains your local fork'\''s integration through Poetry
- Allows direct installation of `docling` without Poetry'\''s dependency resolution
- Both packages can coexist in the same environment

## Key Learnings
1. Poetry'\''s dependency resolution is strict by design for safety
2. The virtual environment can be manipulated directly when needed
3. Sometimes you need to work around package managers for development
4. Local development with forked packages requires special handling

## Best Practices
1. Document when you bypass normal dependency management
2. Test thoroughly when using potentially conflicting packages
3. Consider long-term maintainability when choosing workarounds
4. Keep track of which version of each package is being used

## Future Considerations
- Consider maintaining a forked version of both packages
- Watch for updates to either package that might affect compatibility
- Document any custom modifications that required the local fork