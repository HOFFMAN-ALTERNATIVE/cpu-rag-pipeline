# Git workflow for small teams

## Trunk-based development
Developers integrate small changes into the main branch at least daily, hiding unfinished
work behind feature flags instead of long-lived branches.

## Commit hygiene
Each commit should build and pass tests on its own. Write messages that explain why a change
was made, not only what changed.

## Reverting safely
Prefer `git revert` on shared branches: it adds a new commit that undoes a change without
rewriting history other people have already pulled.
