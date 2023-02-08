How to use the library in Colab
===============================

In order to use the library in Colab, follow these steps.

- Create a GitHub personal access token following [this guide](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token).
    * Set the expiry to e.g. 3 months from now.
    * Tick the 'Full control of private repositories' box.
- Upload [`colab_template.ipynb`](./colab_template.ipynb) to Colab.
- Add you username and personal access token at the top.
- If you run all the cells, it should all work, and import the package at the end.
- You can now change the contents of the file, and add anything you want.

Your Colab notebook isn't actually 'linked' to the GitHub repo. It just clones it and installs it as a package each time you run it.