# Visual algorithm guide (GitHub Pages)

The site is a static visual tutorial at <https://miraflor.github.io/net-center/>
when Pages has been enabled. The site files live under `docs/`; this folder
already contains the package documentation. `.nojekyll` ensures the static
assets are published without a Jekyll build. No npm packages, build step, map
service, remote JavaScript, or backend are needed. No production observations
are used; the worked network is synthetic and measured in projected metres.

## Publish after the website PR passes CI and is merged

1. Open the repository's **Settings → Pages**.
2. Under **Build and deployment**, select **Deploy from a branch**.
3. Select branch **main** and folder **/docs**, then **Save**.
4. Check the Pages deployment in the **Actions** tab. Do not consider the
   website published until that deployment succeeds. Open the URL above.

This repository uses the branch-source publishing option, not a custom deploy
workflow. `.github/workflows/site.yml` validates the site, but does not deploy
it or modify Pages settings. If Pages already uses a different source, review
that configuration before changing it. The site URL assumes the existing
repository name and owner and the default GitHub Pages domain.

## Preview locally

From the repository root:

```bash
python -m http.server 8000 --bind 127.0.0.1 --directory docs
```

Open <http://127.0.0.1:8000/>. Stop the server with Ctrl+C.

## Validate

Node.js 22 is the CI runtime for the visual guide (not a Python runtime dependency).
Run the dependency-free checks from the repository root:

```bash
node --check docs/assets/js/network.js
node --check docs/assets/js/visuals.js
node --check docs/assets/js/site.js
node scripts/test_net_center_site.cjs
```

The JavaScript example enumerates tiny finite candidate sets exactly and
computes the continuous single-center envelope analytically. It is an educational
implementation, not a port of the Python package and not a large-network
benchmark. The tests exercise every weight slider position, both candidate
domains, all four site counts, objective monotonicity, assignments, expected
minimax values, dense edge sampling, static SVG snapshots and internal links.
The Gap statistic and disconnected component allocation are explained but not
executed by the browser. The Python library is the reference for actual work.

When editing the example, update the initial static SVG snapshots and fallback
numbers in `index.html`, then rerun the tests. The initial diagrams and prose
remain useful without JavaScript; interaction and theme selection require it.
Verify the light/dark theme, mobile navigation, the copy buttons, and the
interactive controls in a browser before publishing significant UI changes.
