# How to publish this on GitHub

This directory is already a git repository with one commit. All that remains is
to create the remote and push.

## 1. Create the repository on GitHub

Either through the web UI (https://github.com/new), or with the `gh` CLI after
`gh auth login`:

```bash
gh repo create <repo-name> --public --source . --remote origin --push
```

Use the web UI if `gh` is not installed on the machine you use.

## 2. Or push to an existing empty repository

```bash
git remote add origin https://github.com/<user>/<repo-name>.git
git branch -M main
git push -u origin main
```

## 3. Before the first push

* Fill in the author names in `CITATION.cff` (two `TODO` fields).
* Decide on the licence: `LICENSE` is MIT for the code and `LICENSE-weights`
  applies CC BY 4.0 to `weights/`. Adjust if your institution requires otherwise.
* Confirm that `README.md` Section 8 (licence and citation) matches your choice.

## 4. Getting a DOI for the paper

The data-availability statement needs a repository name and a DOI. GitHub alone
does not issue DOIs. The usual route is:

1. push the repository to GitHub,
2. log in to Zenodo (https://zenodo.org) with your GitHub account,
3. enable the repository in Zenodo's GitHub settings,
4. create a release on GitHub (`git tag v1.0.0 && git push --tags`), which makes
   Zenodo archive that snapshot and mint a DOI.

The DOI then covers exactly the tagged snapshot, which is what a journal expects.
