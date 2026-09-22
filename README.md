# ENGG-Project
Air Pollution prediction model

## Git Workflow (for beginners)

We keep `main` always working. All changes happen on a separate branch, then get merged in through a Pull Request (PR).

1. **Get the latest changes**

   ```bash
   git checkout main
   git pull
   ```

2. **Create a new branch for your work**

   ```bash
   git checkout -b your-name/short-description
   ```

   Example: `git checkout -b baraa/fix-data-cleaning`

3. **Make your changes, then commit them**

   ```bash
   git add .
   git commit -m "Describe what you changed"
   ```

4. **Push your branch to GitHub**

   ```bash
   git push -u origin your-name/short-description
   ```

5. **Open a Pull Request**
   - Go to the repository on GitHub.
   - You'll see a prompt to open a PR for your branch — click it (or open one manually from the "Pull requests" tab).
   - Add a short description of what you changed, then click "Create pull request".

6. **Merge**
   - Once someone reviews and approves the PR (or if working solo, once you've double-checked it), click "Merge pull request" on GitHub.
   - Delete the branch after merging (GitHub will offer a button for this).

7. **Update your local `main`**

   ```bash
   git checkout main
   git pull
   ```

**Rule of thumb:** never commit directly to `main` — always branch, push, and PR.
