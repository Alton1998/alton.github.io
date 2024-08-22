---
draft: false 
date: 2024-08-22
authors:
  - alton
categories:
  - git
  - Interview Questions
---

# How to delete last 5 commits in git?

## Overview
In one of my interview questions I was asked if I knew git, and was given a problem to solve with it. I was asked for a give commit how do you delete the last 5 commits. One thing to not in git there are 100 ways to do the same things so there is no right or wrong way. However, at the end of the day the interviewer decides what is right.

## Solution

Run the following:

```
git reset --hard HEAD~5
```

## References

1. Git - git-reset Documentation. (n.d.). https://git-scm.com/docs/git-reset