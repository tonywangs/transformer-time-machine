/* Completed observations stay on this browser; prompts are never sent elsewhere for storage. */
class PromptHistory {
  constructor() { this.connection = null; }
  open() {
    if (!this.connection) this.connection = new Promise((resolve, reject) => {
      const request = indexedDB.open('ttm-prompt-history', 1);
      request.onupgradeneeded = () => {
        request.result.createObjectStore('runs');
        request.result.createObjectStore('labels', { keyPath: 'key' });
      };
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
    return this.connection;
  }
  async save(size, result) {
    const db = await this.open();
    const key = `${size}:${result.prompt}`;
    return new Promise((resolve, reject) => {
      const tx = db.transaction(['runs', 'labels'], 'readwrite');
      tx.objectStore('runs').put(result, key);
      tx.objectStore('labels').put({ key, size, prompt: result.prompt, savedAt: Date.now() });
      tx.oncomplete = resolve;
      tx.onerror = tx.onabort = () => reject(tx.error);
    });
  }
  async read(store, key) {
    const db = await this.open();
    return new Promise((resolve, reject) => {
      const request = key === undefined ? db.transaction(store).objectStore(store).getAll() : db.transaction(store).objectStore(store).get(key);
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }
  list() { return this.read('labels'); }
  get(key) { return this.read('runs', key); }
}
