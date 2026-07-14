import os
import jwt
import datetime
import bcrypt
import secrets
import logging

logger = logging.getLogger(__name__)
from bson import ObjectId
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings
from tutor_app.utils.db_client import get_collection

@api_view(['POST'])
@permission_classes([AllowAny])
def login_view(request):
    """
    POST /api/auth/login
    Authenticates a student using email and password, updates their active grade selection,
    and returns a stateless HS256 JWT auth token.
    """
    try:
        email = request.data.get('email')
        password = request.data.get('password')
        grade = request.data.get('grade') # Dropdown selected class/grade

        if not email or not password:
            return Response(
                {"error": "Both email and password are required."},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Force input to string format and prevent non-string injection payloads
        if not isinstance(email, str) or not isinstance(password, str):
            return Response(
                {"error": "Email and password must be valid strings."},
                status=status.HTTP_400_BAD_REQUEST
            )

        email_str = email.strip().lower()

        # Query users collection
        users_col = get_collection('users')
        if users_col is None:
            return Response(
                {"error": "Database service is currently unavailable."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        user_doc = users_col.find_one({"email": email_str})
        if not user_doc:
            return Response(
                {"error": "Invalid email or password."},
                status=status.HTTP_401_UNAUTHORIZED
            )

        # Verify password with bcrypt
        db_hash = user_doc.get("password")
        if not db_hash:
            return Response(
                {"error": "Authentication failed due to database hash corruption."},
                status=status.HTTP_401_UNAUTHORIZED
            )

        try:
            if isinstance(db_hash, str):
                db_hash_bytes = db_hash.encode('utf-8')
            else:
                db_hash_bytes = db_hash

            if not bcrypt.checkpw(str(password).encode('utf-8'), db_hash_bytes):
                return Response(
                    {"error": "Invalid email or password."},
                    status=status.HTTP_401_UNAUTHORIZED
                )
        except Exception:
            return Response(
                {"error": "Invalid login credentials format."},
                status=status.HTTP_401_UNAUTHORIZED
            )

        # Check account activity status
        if not user_doc.get("isActive", True):
            return Response(
                {"error": "This account is inactive. Please contact administration."},
                status=status.HTTP_403_FORBIDDEN
            )

        # Student class is locked. Retrieve from database directly and ignore any passed 'grade' param.
        active_grade = user_doc.get("studentClass", {}).get("grade", "8")

        # Generate HS256 stateless Token
        secret_key = os.getenv("DJANGO_SECRET_KEY", settings.SECRET_KEY)
        payload = {
            "userId": str(user_doc["_id"]),
            "email": email_str,
            "grade": active_grade,
            "exp": datetime.datetime.utcnow() + datetime.timedelta(days=7),
            "iat": datetime.datetime.utcnow()
        }

        token = jwt.encode(payload, secret_key, algorithm='HS256')

        return Response({
            "token": token,
            "user": {
                "id": str(user_doc["_id"]),
                "fullName": user_doc.get("fullName"),
                "email": user_doc.get("email"),
                "role": user_doc.get("role", "student"),
                "studentClass": {
                    "grade": active_grade,
                    "section": user_doc.get("studentClass", {}).get("section", ""),
                    "rollNumber": user_doc.get("studentClass", {}).get("rollNumber", "")
                }
            }
        }, status=status.HTTP_200_OK)

    except Exception as e:
        return Response(
            {"error": f"An unexpected error occurred during login verification: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['GET', 'PUT'])
@permission_classes([IsAuthenticated])
def me_view(request):
    """
    GET /api/auth/me - Retrieves the currently authenticated student profile details.
    PUT /api/auth/me - Updates the student profile details (fullName, grade).
    """
    try:
        user = request.user
        
        if request.method == 'PUT':
            # Extract parameters
            full_name = request.data.get('fullName')
            
            users_col = get_collection('users')
            if users_col is None:
                return Response(
                    {"error": "Database service is currently unavailable."},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
                
            update_fields = {}
            if full_name:
                update_fields["fullName"] = str(full_name).strip()
                
            if update_fields:
                users_col.update_one(
                    {"_id": ObjectId(user.id)},
                    {"$set": update_fields}
                )
                
            # Reload updated user document to construct response
            updated_doc = users_col.find_one({"_id": ObjectId(user.id)})
            if not updated_doc:
                return Response(
                    {"error": "Failed to reload updated profile."},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
            
            # Update user representation in request context
            from tutor_app.utils.auth_backend import MongoUser
            user = MongoUser(updated_doc)

        return Response({
            "id": user.id,
            "fullName": user.name,
            "email": user.email,
            "role": user.role,
            "studentClass": {
                "grade": user.grade,
                "section": user.user_doc.get("studentClass", {}).get("section", ""),
                "rollNumber": user.user_doc.get("studentClass", {}).get("rollNumber", "")
            }
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        return Response(
            {"error": f"Failed to handle profile request: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def change_password_view(request):
    """
    POST /api/auth/change-password
    Validates current password and updates it to the new password for logged-in students.
    """
    try:
        user = request.user
        current_password = request.data.get('current_password')
        new_password = request.data.get('new_password')
        
        if not current_password or not new_password:
            return Response(
                {"error": "Both current password and new password are required."},
                status=status.HTTP_400_BAD_REQUEST
            )

        if not isinstance(current_password, str) or not isinstance(new_password, str):
            return Response(
                {"error": "Current password and new password must be valid strings."},
                status=status.HTTP_400_BAD_REQUEST
            )
            
        users_col = get_collection('users')
        if users_col is None:
            return Response(
                {"error": "Database service is currently unavailable."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
            
        user_doc = users_col.find_one({"_id": ObjectId(user.id)})
        if not user_doc:
            return Response(
                {"error": "Student account not found."},
                status=status.HTTP_404_NOT_FOUND
            )
            
        # Verify current password
        db_pwd = user_doc.get("password")
        if not db_pwd or not bcrypt.checkpw(str(current_password).encode('utf-8'), db_pwd.encode('utf-8')):
            return Response(
                {"error": "Incorrect current password."},
                status=status.HTTP_400_BAD_REQUEST
            )
            
        # Hash and save new password
        hashed_password = bcrypt.hashpw(str(new_password).encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        users_col.update_one(
            {"_id": ObjectId(user.id)},
            {"$set": {"password": hashed_password}}
        )
        
        return Response(
            {"message": "Password changed successfully."},
            status=status.HTTP_200_OK
        )
        
    except Exception as e:
        return Response(
            {"error": f"An unexpected error occurred during password change: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def logout_view(request):
    """
    POST /api/auth/logout
    Performs a stateless logout.
    """
    return Response({
        "message": "Successfully logged out. Please discard your authorization token."
    }, status=status.HTTP_200_OK)


@api_view(['POST'])
@permission_classes([AllowAny])
def forgot_password_view(request):
    """
    POST /api/auth/forgot-password
    Generates a 6-digit OTP for resetting password, saves it to student's record with a 15 min expiry,
    and returns a generic success message to prevent user enumeration.
    """
    try:
        email = request.data.get('email')
        if not email:
            return Response(
                {"error": "Email address is required."},
                status=status.HTTP_400_BAD_REQUEST
            )

        if not isinstance(email, str):
            return Response(
                {"error": "Email must be a valid string."},
                status=status.HTTP_400_BAD_REQUEST
            )

        email_str = email.strip().lower()

        users_col = get_collection('users')
        if users_col is None:
            return Response(
                {"error": "Database service is currently unavailable."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        user_doc = users_col.find_one({"email": email_str})
        
        # Generic success message for privacy/security
        success_response = Response(
            {"message": "If an account matches this email, a reset code has been sent."},
            status=status.HTTP_200_OK
        )

        if not user_doc:
            return success_response

        # Generate 6-digit OTP and 15-minute expiration
        otp = str(secrets.SystemRandom().randint(100000, 999999))
        expiry = datetime.datetime.utcnow() + datetime.timedelta(minutes=15)

        users_col.update_one(
            {"_id": user_doc["_id"]},
            {"$set": {
                "resetOtp": otp,
                "resetOtpExpiresAt": expiry.isoformat()
            }}
        )

        logger.info(f"Password reset OTP generated for {email_str}: {otp}")

        return success_response

    except Exception as e:
        return Response(
            {"error": f"An unexpected error occurred during password reset request: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['POST'])
@permission_classes([AllowAny])
def reset_password_view(request):
    """
    POST /api/auth/reset-password
    Validates the 6-digit OTP and updates the student's password using bcrypt.
    """
    try:
        email = request.data.get('email')
        code = request.data.get('code')
        new_password = request.data.get('new_password')

        if not email or not code or not new_password:
            return Response(
                {"error": "Email, verification code, and new password are required."},
                status=status.HTTP_400_BAD_REQUEST
            )

        if not isinstance(email, str) or not isinstance(code, str) or not isinstance(new_password, str):
            return Response(
                {"error": "Email, code, and new password must be valid strings."},
                status=status.HTTP_400_BAD_REQUEST
            )

        email_str = email.strip().lower()
        code_str = code.strip()
        new_password_str = new_password

        users_col = get_collection('users')
        if users_col is None:
            return Response(
                {"error": "Database service is currently unavailable."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        user_doc = users_col.find_one({"email": email_str})
        if not user_doc:
            return Response(
                {"error": "Invalid email or verification code."},
                status=status.HTTP_400_BAD_REQUEST
            )

        db_otp = user_doc.get("resetOtp")
        db_expiry_str = user_doc.get("resetOtpExpiresAt")

        if not db_otp or db_otp != code_str:
            return Response(
                {"error": "Invalid email or verification code."},
                status=status.HTTP_400_BAD_REQUEST
            )

        if not db_expiry_str:
            return Response(
                {"error": "Verification code has expired."},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Parse expiry date
        try:
            db_expiry = datetime.datetime.fromisoformat(db_expiry_str)
        except Exception:
            return Response(
                {"error": "Invalid token state in database."},
                status=status.HTTP_400_BAD_REQUEST
            )

        if datetime.datetime.utcnow() > db_expiry:
            return Response(
                {"error": "Verification code has expired."},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Hash new password
        hashed_password = bcrypt.hashpw(new_password_str.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

        # Update in database and clear reset fields
        users_col.update_one(
            {"_id": user_doc["_id"]},
            {
                "$set": {"password": hashed_password},
                "$unset": {"resetOtp": "", "resetOtpExpiresAt": ""}
            }
        )

        return Response(
            {"message": "Password has been reset successfully."},
            status=status.HTTP_200_OK
        )

    except Exception as e:
        return Response(
            {"error": f"An unexpected error occurred during password reset: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
