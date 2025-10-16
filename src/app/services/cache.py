import hashlib
import json
import time
from typing import Dict, Any, Optional


class ResponseCache:
    def __init__(self, ttl_seconds: int = 3600, max_size: int = 1000):
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size

    def _generate_key(self, request_data: Dict[str, Any]) -> str:
        """Generate a cache key from request data, ignoring non-deterministic fields"""
        # Remove non-deterministic fields for cache key generation
        cache_key_data = request_data.copy()
        if "metadata" in cache_key_data:
            cache_key_data.pop("metadata")
        if "stream" in cache_key_data:
            cache_key_data.pop("stream")

        # Sort the dictionary to ensure consistent keys
        serialized = json.dumps(cache_key_data, sort_keys=True)
        return hashlib.sha256(serialized.encode('utf-8')).hexdigest()

    def get(self, request_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Retrieve a cached response if available and not expired"""
        key = self._generate_key(request_data)
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry['timestamp'] < self.ttl_seconds:
                return entry['response']
            else:
                # Expired entry, remove it
                del self.cache[key]
        return None

    def set(self, request_data: Dict[str, Any], response_data: Dict[str, Any]) -> None:
        """Cache a response"""
        key = self._generate_key(request_data)

        # If cache is full, remove oldest entry
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]['timestamp'])
            del self.cache[oldest_key]

        self.cache[key] = {
            'response': response_data,
            'timestamp': time.time()
        }

    def clear(self) -> None:
        """Clear the cache"""
        self.cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        current_size = len(self.cache)
        current_time = time.time()
        active_entries = sum(1 for entry in self.cache.values()
                             if current_time - entry['timestamp'] < self.ttl_seconds)

        return {
            'size': current_size,
            'max_size': self.max_size,
            'active_entries': active_entries,
            'ttl_seconds': self.ttl_seconds
        }
