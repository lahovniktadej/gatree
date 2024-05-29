# Contributing
Contributions are welcome and greatly appreciated! Please follow the guidelines below to make the process smooth and efficient for everyone involved.

## Table of Contents
- [Contributing](#contributing)
    - [Table of Contents](#table-of-contents)
    - [Types of Contributions](#types-of-contributions)
        - [Bug Reports](#bug-reports)
        - [Feature Requests](#feature-requests)
        - [Bug Fixes](#bug-fixes)
        - [Feature Implementation](#feature-implementation)
        - [Documentation](#documentation)
    - [Pull Request Guidelines](#pull-request-guidelines)
        - [Authoring](#authoring)
        - [Reviewing](#reviewing)
        - [Merging](#merging)
    - [Local Environment for Development](#local-environment-for-development)
        - [Documentation](#documentation-1)
            - [Local Development](#local-development)
            - [Build](#build)
            - [Deployment](#deployment)
        - [Testing](#testing)

## Types of Contributions
### Bug Reports
If you find a bug, please open an issue and tag it with `bug`. The issue should include:
- your operating system name and version,
- GATree version or commit hash,
- a short, descriptive title,
- details steps required to reproduce the bug, and
- details about your local environment that might be helpful in troubleshooting the issue (e.g. Python version, etc.).

Please use [Markdown code blocks](https://help.github.com/articles/creating-and-highlighting-code-blocks) when posting Python stack traces to improve readability.

### Feature Requests
If you'd like to request a new feature, please open an issue and tag it with `enhancement`. The issue should:
- explain the new functionality in detail,
- provide a use case for the new functionality, and
- keep the scope as narrow as possible.

### Bug Fixes
Look through the GitHub issues. Issues tagged with `bug` are open to whoever wants to implement them.

### Feature Implementation
Look through the GitHub issues. Issues tagged with `enhancement` are open to whoever wants to implement them.

### Documentation
GATree could always use more documentation, whether as part of the official GATree docs, in docstrings, or even on the web in blog posts, articles, and such.

## Pull Request Guidelines
### Authoring
- Fill in all sections of the PR template.
- Include a descriptive title.
- Include a summary of the changes and the impact of the changes.
- **Dependencies**: Be mindful about adding new dependencies and avoid unnecessary dependencies.
    - If a new dependency is required, it should be added to `pyproject.toml`.
- **Tests**: The pull request should include tests for the changes made. Ensure that the tests pass before submitting the pull request. See [Testing](#testing) for more information on how to run tests.
- **Documentation**: If the pull request introduces new features or changes existing ones, it should include documentation updates.
- **CI**: Reviewers will not approve a pull request if the CI build fails. You can close and open PR to re-run CI tests.

### Reviewing
- Be constructive when writing reviews.
- Clearly state what needs to be done before the PR can be approved, if there are any required changes.
- The reviewers reserve the right to reject any PR.

### Merging
- The pull request will be merged once it has been approved by at least one reviewer and the CI build passes.
- After the pull request is merged, close the corresponding issues.

## Local Environment for Development
First, [fork the repository on GitHub](https://help.github.com/articles/about-forks). Then, clone your fork locally:
```bash
git clone git@github.com:your-username/gatree.git
cd gatree
```

Next, install the development dependencies using [Poetry](https://python-poetry.org):
```bash
poetry install
```

All of the project's dependencies should be installed and the project should be ready for further development.

#### Dependencies
List of `GATree`'s dependencies:

| Package       | Version | Platform |
|---------------|---------|:--------:|
| scikit-learn  | ^1.3.0  | All      |
| scipy         | ^1.11.2 | All      |
| numpy         | ^1.26.0 | All      |
| pandas        | ^2.1.1  | All      |

List of `GATree`'s development dependencies:

| Package          | Version | Platform |
|------------------|---------|:--------:|
| pytest           | ^7.4.3  | All      |
| sphinx           | ^5.0    | All      |
| sphinx-rtd-theme | ^1.0.0  | All      |

### Documentation
The latest documentation is available at [https://gatree.readthedocs.io](https://gatree.readthedocs.io).

The documentation is built using [Sphinx](https://www.sphinx-doc.org) and hosted on [Read the Docs](https://readthedocs.org). The source files for the documentation are located in the `./docs` directory.

#### Local Development
To set up a local development environment, install the optional dependencies:
```bash
poetry install --extras docs
```

#### Build
To build the documentation, run the following command:
```bash
poetry run sphinx-build ./docs ./docs/build
```

#### Deployment
Commits to `master` will automatically deploy the documentation to [Read the Docs](https://gatree.readthedocs.io).

### Testing
All tests are carried out using the [pytest](https://docs.pytest.org) testing framework. The source files for the tests are located in the `./tests` directory.

To run all tests, use the following command:
```bash
poetry run pytest
```

To run a specific test, use the following command:
```bash
poetry run pytest tests/test_file.py
```