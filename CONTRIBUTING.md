Guide to contributing code
==========================

First steps
-----------

- Fork the repo.

- Clone your fork:
```
git clone git@github.com:USERNAME/al-llm-experiments.git
```

- Add the the upstream remote:
```
git remote add upstream git@github.com:OxAI-Safety-Hub/al-llm-experiments.git
```

- Create a Python virtual environment.

- Install the requirements:
```
pip install -r requirements.txt
```


Contributing a pull request
---------------------------

- Synchronise your `main` branch `upstream/main`:
```
git checkout main
git fetch upstream
git merge upstream/main
```

- Create a new branch for your feature:
```
git checkout -b new_feature
```

- Make some modifications, following the [Coding Guidelines](#coding-guidelines) below.

- Run `black` to format the code nicely:
```
black .
```

- Make sure there are no [PEP-8](https://peps.python.org/pep-0008/) violations:
```
flake8 .
```

- Commit your changes:
```
git add files
git commit -m "Commit message"
```

- Synchronise your `main` branch `upstream/main` again:
```
git checkout main
git fetch upstream
git merge upstream/main
```

- Go back to your feature branch:
```
git checkout new_feature
```

- If there are any changes to the `main` branch, merge them into your feature branch:
```
git merge main
```

- Push your branch to your fork on GitHub:
```
git push origin new_feature
```

- Create a pull request on GitHub.


Coding Guidelines
-----------------

- Follow [PEP-8](https://peps.python.org/pep-0008/).
- Use the [Numpy Docstring Standard](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard) for docstrings.


Working in VS Code
------------------

- Install the [Python extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python).
- Set the `python.formatting.provider` setting to `black` for your workspace/folder.
- This allows you to format the code within VS Code.
- Install the [GitLens extension](https://marketplace.visualstudio.com/items?itemName=eamodio.gitlens).
- This allows you to do all the git commands listed above through the sidebar.


Acknowledgements
----------------

Partly based on the [Scikit-Learn contributing guide](https://scikit-learn.org/dev/developers/contributing.html).