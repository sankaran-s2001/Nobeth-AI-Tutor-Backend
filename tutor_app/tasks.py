import logging
import datetime
from celery import shared_task
from bson import ObjectId
from tutor_app.utils.db_client import get_collection

logger = logging.getLogger(__name__)

@shared_task
def log_study_time_async(user_id_str, duration_minutes, activity_type):
    """
    Celery task: Asynchronously logs study time in MongoDB study_logs.
    """
    try:
        logs_col = get_collection('study_logs')
        if logs_col is None:
            logger.error("Celery task: Database connection is offline.")
            return False

        today_str = datetime.date.today().strftime("%Y-%m-%d")

        # Increment minutes for today's entry
        logs_col.update_one(
            {
                "userId": ObjectId(user_id_str),
                "date": today_str
            },
            {
                "$inc": {"duration_minutes": duration_minutes},
                "$setOnInsert": {
                    "activity_type": activity_type,
                    "created_at": datetime.datetime.utcnow().isoformat()
                }
            },
            upsert=True
        )
        logger.info(f"Celery task: Successfully logged {duration_minutes}m study time for user {user_id_str}")
        return True
    except Exception as e:
        logger.error(f"Celery task: Failed to write study duration: {e}")
        return False
