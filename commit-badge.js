// Small, faint 7-character commit-hash badge pinned to the top-right corner of
// every page. There is no build step to stamp the SHA in, so the deployed
// commit is read at runtime from the GitHub API — it always reflects whatever
// GitHub Pages last built from `main`. Fails silently (badge stays hidden) if
// the API is unreachable or rate-limited.
(function () {
  'use strict';

  var REPO = 'aeriab/Bacterial_Genome_Sweep_Scan_Tool';

  var badge = document.createElement('span');
  badge.id = 'commit-badge';
  badge.className = 'commit-badge';
  badge.hidden = true;

  function attach() { document.body.appendChild(badge); }
  if (document.body) attach();
  else document.addEventListener('DOMContentLoaded', attach);

  fetch('https://api.github.com/repos/' + REPO + '/commits/main', {
    headers: { Accept: 'application/vnd.github+json' },
  })
    .then(function (r) { return r.ok ? r.json() : Promise.reject(new Error(r.status)); })
    .then(function (data) {
      if (!data || typeof data.sha !== 'string') return;
      badge.textContent = data.sha.slice(0, 7);
      badge.title = 'deployed commit ' + data.sha.slice(0, 7);
      badge.hidden = false;
    })
    .catch(function () { /* leave the badge hidden */ });
})();
