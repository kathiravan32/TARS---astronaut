/*
 * TARS offline service worker.
 *
 * What this does: the first time you load task_monitor.html WITH internet,
 * every file it fetches (the TensorFlow.js library, coco-ssd, pose-detection,
 * hand-pose-detection, and — importantly — the actual AI model weight files
 * those libraries download at runtime) gets copied into this cache.
 *
 * Every load after that is served from the cache first, so the page keeps
 * working with zero internet connection — airplane mode, no wifi, doesn't
 * matter — as long as you don't clear this browser's site data for this page.
 *
 * You do NOT need to list every model file by hand: this caches whatever the
 * page actually requests, generically, for the CDN hosts listed below.
 */

const CACHE_NAME = 'tars-offline-v1';

// Hosts that TensorFlow.js / coco-ssd / pose-detection / hand-pose-detection
// actually fetch from at runtime (library code + model weight shards).
const CACHEABLE_HOSTS = [
  'cdn.jsdelivr.net',
  'storage.googleapis.com',
  'tfhub.dev',
  'www.gstatic.com',
  'raw.githubusercontent.com'
];

self.addEventListener('install', (event) => {
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET') return;

  const url = new URL(req.url);
  const sameOrigin = url.origin === self.location.origin;
  const cacheableHost = CACHEABLE_HOSTS.includes(url.hostname);
  if (!sameOrigin && !cacheableHost) return; // let it pass through untouched

  event.respondWith(
    caches.open(CACHE_NAME).then(async (cache) => {
      const cached = await cache.match(req);
      if (cached) return cached; // instant, offline-safe

      try {
        const response = await fetch(req);
        // Only cache genuinely successful responses (avoid caching errors)
        if (response && (response.status === 200 || response.type === 'opaque')) {
          cache.put(req, response.clone());
        }
        return response;
      } catch (err) {
        // Offline and not cached yet (e.g. first-ever load with no internet) —
        // nothing we can do for this specific file.
        if (cached) return cached;
        throw err;
      }
    })
  );
});
