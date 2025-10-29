"""
Production Dragonfly DB Configuration
Azure Australia East - REAL connection details from environment
"""

import redis
import os
from typing import Optional, Dict, Any
import json

class DragonflyConfig:
    """Production Dragonfly Database Configuration"""
    
    HOST = os.getenv("DRAGONFLY_HOST", "zy0n0f6e8.dragonflydb.cloud")
    PORT = int(os.getenv("DRAGONFLY_PORT", "6385"))
    PASSWORD = os.getenv("DRAGONFLY_PASSWORD", "")
    
    @classmethod
    def get_connection_uri(cls) -> str:
        """Build connection URI from environment variables"""
        if not cls.PASSWORD:
            raise ValueError("DRAGONFLY_PASSWORD environment variable not set")
        return f"rediss://default:{cls.PASSWORD}@{cls.HOST}:{cls.PORT}"
    
    PROVIDER = "Azure"
    REGION = "Australia East"
    CLUSTER_MODE = "Emulated (Single Shard)"
    PLAN = "50 GB (Extreme)"
    VERSION = "v1.33.1"
    
    TLS_ENABLED = True
    ENDPOINT_TYPE = "Public"
    EVICTION_POLICY = "No Eviction"
    MASTER_ZONE = 1

class DragonflyClient:
    """Production Dragonfly Redis Client"""
    
    def __init__(self):
        self.client: redis.Redis
        self._connect()
    
    def _connect(self):
        """Establish connection to Dragonfly DB"""
        try:
            connection_uri = DragonflyConfig.get_connection_uri()
            
            self.client = redis.from_url(
                connection_uri,
                decode_responses=True,
                ssl_cert_reqs=None,
                socket_keepalive=True,
                socket_connect_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            self.client.ping()
            print(f"✅ Connected to Dragonfly DB - {DragonflyConfig.REGION}")
            
        except Exception as e:
            print(f"❌ Dragonfly connection failed: {e}")
            raise
    
    def save_training_metadata(self, epoch: int, loss: float, gpu_count: int, timestamp: str):
        """Save training metadata to Dragonfly"""
        key = f"training:epoch:{epoch}"
        data = {
            "epoch": str(epoch),
            "loss": str(loss),
            "gpu_count": str(gpu_count),
            "timestamp": timestamp,
            "region": DragonflyConfig.REGION
        }
        
        self.client.hset(key, mapping=data)  # type: ignore
        self.client.zadd("training:timeline", {key: epoch})
        
        print(f"💾 Saved to Dragonfly: epoch {epoch}, loss {loss}")
    
    def save_model_checkpoint(self, epoch: int, model_path: str, metadata: dict):
        """Save model checkpoint reference to Dragonfly"""
        key = f"model:checkpoint:{epoch}"
        
        checkpoint_data = {
            "epoch": str(epoch),
            "model_path": model_path,
            "metadata": json.dumps(metadata),
            "stored_at": DragonflyConfig.REGION
        }
        
        self.client.hset(key, mapping=checkpoint_data)  # type: ignore
        self.client.lpush("model:checkpoints", key)
        
        print(f"✅ Model checkpoint {epoch} saved to Dragonfly")
    
    def get_latest_checkpoint(self) -> Optional[dict]:
        """Get latest model checkpoint from Dragonfly"""
        checkpoint_key = self.client.lindex("model:checkpoints", 0)
        
        if checkpoint_key:
            data = self.client.hgetall(checkpoint_key)
            if data:
                data['metadata'] = json.loads(data.get('metadata', '{}'))
                return data
        
        return None
    
    def save_training_metrics(self, metrics: Dict[str, Any]):
        """Save comprehensive training metrics"""
        key = f"metrics:{metrics['timestamp']}"
        self.client.hset(key, mapping=metrics)  # type: ignore
        self.client.expire(key, 86400 * 30)
    
    def close(self):
        """Close Dragonfly connection"""
        if self.client:
            self.client.close()
            print("🔌 Dragonfly connection closed")
