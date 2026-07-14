import os
import logging
from pymongo import MongoClient
from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)

# Env credentials
MONGODB_URI = os.getenv("MONGODB_URI")
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Singleton clients loaded lazily
_mongo_client = None
_db = None
_qdrant_client = None

def get_mongo_client():
    """Lazily initializes and pings the MongoDB client connection."""
    global _mongo_client
    if _mongo_client is None:
        if not MONGODB_URI:
            logger.error("MONGODB_URI environment variable is missing.")
            return None
        try:
            logger.info("Initializing MongoDB Client connection...")
            # Set connection timeout to 5 seconds
            _mongo_client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
            _mongo_client.admin.command('ping')
            logger.info("Successfully connected to MongoDB Atlas.")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB Atlas: {e}")
            _mongo_client = None
    return _mongo_client

def get_db():
    """Returns the default MongoDB database reference."""
    global _db
    if _db is None:
        client = get_mongo_client()
        if client:
            try:
                # Retrieve the database name specified in the URI connection string
                db_name = client.get_default_database().name
            except Exception:
                db_name = 'test'
            _db = client[db_name]
            
            # Setup indexes for maximum query performance in production
            try:
                _db['users'].create_index('email', unique=True)
                _db['chats'].create_index([('userId', 1), ('pinned', -1), ('updatedAt', -1)])
                _db['chats'].create_index('sessionId')
                _db['study_logs'].create_index([('userId', 1), ('date', 1)])
                _db['study_goals'].create_index('userId', unique=True)
                _db['svg_diagrams'].create_index('title')
                _db['imagemetadatas'].create_index('title')
                _db['query_cache'].create_index('key', unique=True)
                _db['query_cache'].create_index('created_at', expireAfterSeconds=86400)
            except Exception as e:
                logger.error(f"Failed to create collection indexes: {e}")
    return _db

def get_qdrant_client():
    """Lazily initializes the Qdrant Cloud Client instance."""
    global _qdrant_client
    if _qdrant_client is None:
        if not QDRANT_HOST:
            logger.error("QDRANT_HOST environment variable is missing.")
            return None
        try:
            logger.info("Initializing Qdrant Cloud Client...")
            _qdrant_client = QdrantClient(
                url=QDRANT_HOST,
                api_key=QDRANT_API_KEY
            )
            logger.info("Successfully connected to Qdrant Cloud.")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant Cloud Client: {e}")
            _qdrant_client = None
    return _qdrant_client

def get_collection(name):
    """Utility to quickly grab collections from the MongoDB client."""
    database = get_db()
    if database is not None:
        return database[name]
    return None
