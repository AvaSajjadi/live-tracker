# How we work together

## The golden rules
1. Never push directly to `main`. Always use a branch and a pull request (PR).
2. Every PR needs one teammate to look at it before merging.
3. Tests must pass (green check on the PR) before merging.
4. No data, videos, or API keys in Git. Data lives in the shared Google Drive folder.
5. One task = one GitHub Issue with one owner.

## Daily routine (coders)

Start of the day, get everyone's latest work:
```powershell
git checkout main
git pull
git checkout -b my-task-name      # e.g. calms21-loader
```

While working, save often:
```powershell
git add <files you changed>
git commit -m "Short description of what you did"
```

When done (or at the end of the day):
```powershell
.\.venv\Scripts\python.exe -m pytest -q    # make sure tests pass
git push -u origin my-task-name
```
Then on GitHub: open a Pull Request, fill in the template, ask a teammate to review.

If `main` changed while you worked:
```powershell
git checkout main
git pull
git checkout my-task-name
git merge main
```
If Git says there is a "merge conflict", don't panic: ask in the group chat.

## Non-coders
- Edit Markdown docs (README, INTERFACES.md) directly on GitHub with the pencil icon,
  choose "Create a new branch", then open a pull request.
- Slides, the Word report, and notes go in the shared Google Drive folder.
- Comment on Issues and PRs to give feedback on how behaviors are defined.

## First-time setup (Windows PowerShell)
```powershell
git clone https://github.com/<owner>/live-tracker.git
cd live-tracker
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e . -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
copy .env.example .env      # then add your API key inside .env (optional)
```

## Daily check-in (10 min, all six)
Each person: what I finished, what I'm doing next, what's blocking me.
