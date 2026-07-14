import jwt
import os
import logging
from bson import ObjectId
from rest_framework import authentication
from rest_framework import exceptions
from django.conf import settings
from tutor_app.utils.db_client import get_collection

logger = logging.getLogger(__name__)

class MongoUser:
    """
    Wrapper class representing the authenticated student,
    conforming to Django's standard User interface requirements.
    """
    def __init__(self, user_doc):
        self.user_doc = user_doc

    @property
    def is_authenticated(self):
        return True

    @property
    def is_anonymous(self):
        return False

    @property
    def id(self):
        return str(self.user_doc.get("_id"))

    @property
    def email(self):
        return self.user_doc.get("email")

    @property
    def name(self):
        return self.user_doc.get("fullName")

    @property
    def role(self):
        return self.user_doc.get("role", "student")

    @property
    def grade(self):
        # Extract studentClass grade if present
        student_class = self.user_doc.get("studentClass", {})
        return student_class.get("grade") if isinstance(student_class, dict) else None


class MongoTokenAuthentication(authentication.BaseAuthentication):
    """
    Custom Stateless Token-based Authentication for MongoDB User Documents.
    Decodes HS256 JWT tokens passed via Authorization: Bearer <token>.
    """
    def authenticate(self, request):
        auth_header = request.headers.get('Authorization') or request.META.get('HTTP_AUTHORIZATION')
        if not auth_header:
            return None

        # Exclude other formats and extract token
        parts = auth_header.split()
        if len(parts) != 2 or parts[0].lower() != 'bearer':
            return None

        token = parts[1]
        secret_key = os.getenv("DJANGO_SECRET_KEY", settings.SECRET_KEY)

        try:
            # Decode the stateless token
            payload = jwt.decode(token, secret_key, algorithms=['HS256'])
        except jwt.ExpiredSignatureError:
            raise exceptions.AuthenticationFailed("Authorization token has expired.")
        except jwt.InvalidTokenError:
            raise exceptions.AuthenticationFailed("Invalid authorization token.")

        user_id = payload.get("userId")
        if not user_id:
            raise exceptions.AuthenticationFailed("Malformed token payload.")

        # Convert to ObjectId safely
        try:
            obj_id = ObjectId(user_id)
        except Exception:
            raise exceptions.AuthenticationFailed("Invalid student ID format in token.")

        # Query MongoDB
        try:
            users_col = get_collection('users')
            if users_col is None:
                raise exceptions.AuthenticationFailed("Database service is currently unavailable.")
                
            user_doc = users_col.find_one({"_id": obj_id})
        except Exception as e:
            logger.error(f"Database query error during authentication: {e}")
            raise exceptions.AuthenticationFailed("Error checking student credentials.")

        if not user_doc:
            raise exceptions.AuthenticationFailed("Student account not found.")

        if not user_doc.get("isActive", True):
            raise exceptions.AuthenticationFailed("Student account is inactive.")

        # Wrap in MongoUser
        mongo_user = MongoUser(user_doc)
        
        # Return tuple (user, auth)
        return (mongo_user, token)
