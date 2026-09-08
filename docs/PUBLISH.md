# Publishing Starfelt

1. PyPI → Publishing → Trusted publishers → GitHub  
   - Owner: `victorachede`  
   - Repo: `starfelt`  
   - Workflow: `publish.yml`  
   - Environment: (optional)

2. Tag a release:

```bash
git checkout main
git pull
git tag v0.1.0
git push origin v0.1.0
```

3. GitHub Action **Publish to PyPI** builds and uploads.

Until the publisher is linked, the publish job will fail at the upload step — CI tests still run on every PR.
