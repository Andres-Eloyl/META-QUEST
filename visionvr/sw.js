const CACHE_NAME = 'visionvr-v1';
const urlsToCache = [
  '/',
  '/index.html',
  '/js/aframe.min.js',
  '/js/socket.io.min.js'
];

self.addEventListener('install', event => {
  // Realizar la instalación del Service Worker y cachear archivos estáticos
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => {
        console.log('Archivos cacheados correctamente');
        return cache.addAll(urlsToCache);
      })
  );
});

self.addEventListener('fetch', event => {
  // Ignorar las peticiones al Socket o al backend /detectar
  if (event.request.url.includes('/socket.io') || event.request.url.includes('/detectar')) {
    return;
  }
  
  event.respondWith(
    caches.match(event.request)
      .then(response => {
        // Devolver respuesta cacheada si existe
        if (response) {
          return response;
        }
        return fetch(event.request);
      })
  );
});

self.addEventListener('activate', event => {
  // Limpiar cachés antiguos si se actualiza la versión
  const cacheAllowlist = [CACHE_NAME];
  event.waitUntil(
    caches.keys().then(cacheNames => {
      return Promise.all(
        cacheNames.map(cacheName => {
          if (cacheAllowlist.indexOf(cacheName) === -1) {
            return caches.delete(cacheName);
          }
        })
      );
    })
  );
});
