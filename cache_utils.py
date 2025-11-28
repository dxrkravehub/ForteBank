import redis
import hashlib
import json
from typing import Optional, Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()

try:
    redis_client = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        db=0,
        decode_responses=True,
        socket_connect_timeout=2,
        socket_timeout=2
    )
    # Проверка подключения
    redis_client.ping()
    REDIS_AVAILABLE = True
    print("Redis connected successfully")
except Exception as e:
    REDIS_AVAILABLE = False
    redis_client = None
    print(f" Redis unavailable: {e}")
    print("   Caching disabled, system will work without cache")


def get_cache_key(query: str, cache_type: str = "verdict") -> str:
    """
    Генерирует уникальный хеш-ключ для кеширования.
    """
    content = f"{cache_type}:{query.lower().strip()}".encode('utf-8')
    hash_digest = hashlib.sha256(content).hexdigest()[:16]
    return f"procurement:{cache_type}:{hash_digest}"


def get_cached_result(query: str, cache_type: str = "verdict") -> Optional[Dict[str, Any]]:
    """
    Получает закешированный результат из Redis.
    """
    if not REDIS_AVAILABLE:
        return None
    
    try:
        key = get_cache_key(query, cache_type)
        cached = redis_client.get(key)
        
        if cached:
            print(f"Cache HIT: {query[:50]}...")
            result = json.loads(cached)
            # Добавляем метаданные о том, что это из кеша
            result['_from_cache'] = True
            return result
        
        print(f"Cache MISS: {query[:50]}...")
        return None
    
    except json.JSONDecodeError as e:
        print(f"Cache JSON decode error: {e}")
        # Удаляем битый кеш
        redis_client.delete(key)
        return None
    
    except Exception as e:
        print(f"⚠️ Redis get error: {e}")
        return None

def cache_result(
    query: str, 
    result: Dict[str, Any], 
    cache_type: str = "verdict", 
    ttl: int = 86400
) -> bool:
    """
    Сохраняет результат в Redis с указанным TTL.

    Example:
        verdict = {"tender_id": "123", "risk_score": 85}
        cache_result("серверы Dell", verdict, ttl=3600)
        Cached: серверы Dell (TTL: 3600s)
        True
    """
    if not REDIS_AVAILABLE:
        return False
    
    try:
        key = get_cache_key(query, cache_type)
        
        # Убираем служебные метаданные перед сохранением
        clean_result = {k: v for k, v in result.items() if not k.startswith('_')}
        
        redis_client.setex(
            key,
            ttl,
            json.dumps(clean_result, ensure_ascii=False)
        )
        
        print(f" Cached: {query[:50]}... (TTL: {ttl}s)")
        return True
    
    except TypeError as e:
        print(f"Cache save error - data not JSON serializable: {e}")
        return False
    
    except Exception as e:
        print(f" Cache save error: {e}")
        return False


def clear_cache(pattern: str = "procurement:*") -> int:
    """
    Очищает кеш по указанному паттерну.
    
    Example:
        clear_cache("procurement:verdict:*")
        Cleared 15 cache entries
        15
    """
    if not REDIS_AVAILABLE:
        print("Redis unavailable, nothing to clear")
        return 0
    
    try:
        keys = list(redis_client.scan_iter(pattern))
        
        if keys:
            deleted = redis_client.delete(*keys)
            print(f"Cleared {deleted} cache entries")
            return deleted
        
        print("ℹ️ No cache entries found")
        return 0
    
    except Exception as e:
        print(f"ache clear error: {e}")
        return 0


def get_cache_stats() -> Dict[str, Any]:
    if not REDIS_AVAILABLE:
        return {
            "available": False,
            "message": "Redis is not available"
        }
    
    try:
        info = redis_client.info("stats")
        keys_count = redis_client.dbsize()
        
        hits = info.get('keyspace_hits', 0)
        misses = info.get('keyspace_misses', 1)
        total_requests = hits + misses
        hit_rate = (hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "available": True,
            "total_keys": keys_count,
            "hits": hits,
            "misses": misses,
            "hit_rate": round(hit_rate, 2),
            "memory_used": info.get('used_memory_human', 'N/A'),
            "uptime_days": info.get('uptime_in_days', 0)
        }
    
    except Exception as e:
        return {
            "available": False,
            "error": str(e)
        }


def invalidate_old_cache(max_age_hours: int = 168) -> int:
    """
    Удаляет кеш старше указанного времени (default: 7 дней).
    
    Args:
        max_age_hours: Максимальный возраст кеша в часах
    
    Returns:
        int: Количество удалённых записей
    """
    if not REDIS_AVAILABLE:
        return 0
    
    try:
        deleted_count = 0
        max_age_seconds = max_age_hours * 3600
        
        for key in redis_client.scan_iter("procurement:*"):
            ttl = redis_client.ttl(key)
            
            # Если TTL меньше оставшегося времени до max_age
            if ttl != -1 and (86400 - ttl) > max_age_seconds:
                redis_client.delete(key)
                deleted_count += 1
        
        if deleted_count > 0:
            print(f" Cleaned up {deleted_count} old cache entries")
        
        return deleted_count
    
    except Exception as e:
        print(f"Invalidation error: {e}")
        return 0


def is_redis_available() -> bool:
    """Проверяет доступность Redis."""
    return REDIS_AVAILABLE


if __name__ == "__main__":
    print("\n" + "="*60)
    print("REDIS CACHE TESTING")
    print("="*60)
    
    print("\n[TEST 1] Save & Retrieve")
    test_verdict = {
        "tender_id": "TEST-001",
        "risk_score": 75,
        "verdict": "HIGH_RISK"
    }
    
    query = "тестовый запрос серверы"
    cache_result(query, test_verdict, ttl=60)
    
    retrieved = get_cached_result(query)
    if retrieved:
        print(f"Retrieved: {retrieved}")
    
    print("\n[TEST 2] Cache Statistics")
    stats = get_cache_stats()
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    
    print("\n[TEST 3] Clear Cache")
    cleared = clear_cache("procurement:verdict:*")
    print(f"Cleared {cleared} entries")
    
    print("\n" + "="*60)
    print("TESTING COMPLETE")
    print("="*60)