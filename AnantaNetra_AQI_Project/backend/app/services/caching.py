import json
import logging
from typing import Any, Optional
from datetime import timedelta

logger = logging.getLogger(__name__)

class CacheManager:
    def __init__(self):
        try:
            import redis
            from app.utils.config import settings
            self.redis_client = redis.from_url(settings.redis_url)
            self.redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Using in-memory cache.")
            self.redis_client = None
            self._memory_cache = {}
    
    async def get(self, key: str) -> Optional[Any]:
        """Get cached value"""
        try:
            if self.redis_client:
                cached_value = self.redis_client.get(key)
                if cached_value:
                    return json.loads(cached_value)
            else:
                return self._memory_cache.get(key)
        except Exception as e:
            logger.error(f"Cache get error: {e}")
        return None
    
    async def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set cached value"""
        try:
            from app.utils.config import settings
            if self.redis_client:
                return self.redis_client.setex(
                    key, 
                    ttl or settings.cache_ttl, 
                    json.dumps(value, default=str)
                )
            else:
                self._memory_cache[key] = value
                return True
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete cached value"""
        try:
            if self.redis_client:
                return bool(self.redis_client.delete(key))
            else:
                return bool(self._memory_cache.pop(key, None))
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False

cache_manager = CacheManager()
