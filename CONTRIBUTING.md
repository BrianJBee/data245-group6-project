
# Project Workflow

This document outlines the basic workflow for setting up the repository, making changes, and submitting contributions.

**These are just guidelines,** do not waste time on conventions if time does not permit. Focus on pushing necessary changes and results before the deadlines.

## Getting Started

### Prerequisites

Before contributing, make sure you have [Git](https://git-scm.com/) installed and any project-specific dependencies listed in [README.md](README.md).

### Clone the Repository

Clone the repository to your local machine:

```bash
git clone https://github.com/BrianJBee/data245-group6-project.git
cd data245-group6-project.git
```

Follow any additional setup instructions in [README.md](README.md).

### Install nbstripout

This project uses [nbstripout](https://github.com/kynan/nbstripout) to remove outputs from Jupyter notebooks before they are committed. This helps keep notebook diffs readable and prevents large output files from cluttering the repository.

Install it with conda (make sure you've activated the project environment):

```bash
conda install nbstripout
```

Configure it for this repository:

```bash
nbstripout --install
```

This installs a Git filter in your local repository that strips notebook outputs when you commit changes. You only need to configure it once per local clone.

Please avoid committing unnecessary notebook outputs. If outputs are needed to illustrate a result, discuss in the discord group chat so we can decide where to put it. We will likely include it in our presentation or report.

## Development Workflow

We use a standard Git workflow: create a branch for each contribution, make changes on that branch, and submit a pull request for review.

### 1. Update Your Local Repository

Before starting new work, make sure your local copy is up to date:

```bash
git checkout main
git pull origin main
```

Replace `main` with the default branch name if necessary.

### 2. Create a Branch

Create a new branch for each feature, fix, or other contribution. Branch names should follow this convention:

`<name>/<feature>`

For example:

```bash
git checkout -b andrea/data-cleaning
git checkout -b brian/chores
```

Use your name or username before the slash and a short, descriptive feature name after it. Use lowercase letters and hyphens where appropriate.

### 3. Make Your Changes

Make changes on your feature branch rather than directly on `main`.

- Keep changes focused on a single task when practical.
- Follow the existing code style and project conventions.
- Update documentation or tests when relevant.
- Avoid committing temporary files or unrelated changes.
- Keep Jupyter notebooks clean of unnecessary outputs.

### 4. Commit Your Changes

Review your changes before committing:

```bash
git status
git diff
```

Stage and commit the relevant files:

```bash
git add <file>
git commit -m "Describe your changes"
```

Write clear, concise commit messages that summarize the changes. For example:

- `feat: add data preprocessing script`
- `fix: adjust figure axis labels`
- `docs: update experiment documentation`

If `nbstripout` is configured correctly, notebook outputs will be stripped automatically when you commit. You can still review the staged changes before committing to make sure the diff is as expected.

### 5. Push Your Branch

Push your branch to the remote repository:

```bash
git push -u origin <name>/<feature>
```

After the initial push, you can generally use `git push` for subsequent updates.

### 6. Open a Pull Request

When your changes are ready for review:

1. [Open a pull request (PR)](https://github.com/BrianJBee/data245-group6-project/pulls) from your feature branch into `main`.
2. Provide a brief description of the changes and their motivation.
3. Include relevant context, results, figures, or references when applicable.
4. Link related issues or discussions, if applicable.
5. Respond to review feedback and make any requested changes.

Keep pull requests focused and provide enough context for others to understand and evaluate your contribution.

## Keeping Your Branch Up to Date

If `main` has changed while you are working, update your branch as needed:

```bash
git checkout main
git pull origin main
git checkout <name>/<feature>
git merge main
```

Resolve any merge conflicts, review the resulting changes, and run relevant tests or checks before pushing again.

Alternatively, contributors familiar with rebasing may use `git rebase main`. Avoid rebasing shared branches without coordinating with other contributors.

## Before Submitting

Before opening or updating a pull request, check that:

- [ ] Your branch follows the `<name>/<feature>` naming convention.
- [ ] Your changes are limited to the intended task.
- [ ] Relevant tests, scripts, or analyses run as expected.
- [ ] Documentation is updated where necessary.
- [ ] Notebook outputs are stripped where appropriate.
- [ ] No unintended files, credentials, or sensitive data are included.
- [ ] Your commits have clear, descriptive messages.

Not every contribution will require tests or documentation changes; use your judgment based on the scope of the work.

## Questions and Discussion

If you have questions, encounter problems, or want feedback on a proposed change, open an issue or discuss in the discord group chat.

For larger changes, discussing the proposed approach before implementation can help avoid duplicated effort and improve collaboration.
