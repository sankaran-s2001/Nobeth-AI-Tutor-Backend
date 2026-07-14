import jwt
import bcrypt
import datetime
import uuid
from django.test import TestCase
from django.conf import settings
from unittest.mock import MagicMock, patch
from bson import ObjectId
from rest_framework import status

# Import target modules for monkey-patching
import tutor_app.utils.auth_backend
import tutor_app.views.auth
import tutor_app.views.dashboard
import tutor_app.views.chat
import tutor_app.tasks
import tutor_app.utils.rag_service
import tutor_app.views.content

class NobethTutorAPITests(TestCase):

    def setUp(self):
        # Store original functions for clean restoration
        self.orig_auth_get = tutor_app.utils.auth_backend.get_collection
        self.orig_auth_view_get = tutor_app.views.auth.get_collection
        self.orig_dash_get = tutor_app.views.dashboard.get_collection
        self.orig_chat_get = tutor_app.views.chat.get_collection
        self.orig_tasks_get = tutor_app.tasks.get_collection
        self.orig_rag_get = tutor_app.utils.rag_service.get_collection
        self.orig_rag_qdrant = tutor_app.utils.rag_service.get_qdrant_client
        self.orig_content_get = tutor_app.views.content.get_collection

        # Initialize global collection mock dictionary
        self.collections = {}

        def mock_get_collection(name):
            if name not in self.collections:
                self.collections[name] = MagicMock()
            return self.collections[name]

        # Apply monkey-patches
        tutor_app.utils.auth_backend.get_collection = mock_get_collection
        tutor_app.views.auth.get_collection = mock_get_collection
        tutor_app.views.dashboard.get_collection = mock_get_collection
        tutor_app.views.chat.get_collection = mock_get_collection
        tutor_app.tasks.get_collection = mock_get_collection
        tutor_app.utils.rag_service.get_collection = mock_get_collection
        tutor_app.views.content.get_collection = mock_get_collection

        # Sample mock user document
        self.user_id = ObjectId()
        self.mock_user = {
            "_id": self.user_id,
            "fullName": "Sankaran S",
            "email": "sankaran@gmail.com",
            "password": bcrypt.hashpw("securepass".encode('utf-8'), bcrypt.gensalt()).decode('utf-8'),
            "role": "student",
            "studentClass": {"grade": "8"},
            "isActive": True
        }

        # Seed mock user in the users mock collection
        self.collections['users'] = MagicMock()
        self.collections['users'].find_one.return_value = self.mock_user

        # Generate a valid JWT token for authenticated endpoints
        self.token_payload = {
            "userId": str(self.user_id),
            "email": "sankaran@gmail.com",
            "exp": datetime.datetime.utcnow() + datetime.timedelta(days=1)
        }

        import os
        secret_key = os.getenv("DJANGO_SECRET_KEY", settings.SECRET_KEY)
        self.token = jwt.encode(self.token_payload, secret_key, algorithm="HS256")
        self.auth_headers = {
            "HTTP_AUTHORIZATION": f"Bearer {self.token}"
        }

    def tearDown(self):
        # Restore original functions to avoid side-effects on other test modules
        tutor_app.utils.auth_backend.get_collection = self.orig_auth_get
        tutor_app.views.auth.get_collection = self.orig_auth_view_get
        tutor_app.views.dashboard.get_collection = self.orig_dash_get
        tutor_app.views.chat.get_collection = self.orig_chat_get
        tutor_app.tasks.get_collection = self.orig_tasks_get
        tutor_app.utils.rag_service.get_collection = self.orig_rag_get
        tutor_app.utils.rag_service.get_qdrant_client = self.orig_rag_qdrant
        tutor_app.views.content.get_collection = self.orig_content_get

    def test_login_success(self):
        payload = {
            "email": "sankaran@gmail.com",
            "password": "securepass",
            "grade": "8"
        }
        response = self.client.post('/api/auth/login', payload, content_type='application/json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("token", response.data)
        self.assertEqual(response.data["user"]["fullName"], "Sankaran S")
        self.assertEqual(response.data["user"]["studentClass"]["grade"], "8")

    def test_login_invalid_credentials(self):
        payload = {
            "email": "sankaran@gmail.com",
            "password": "wrongpassword",
            "grade": "8"
        }
        response = self.client.post('/api/auth/login', payload, content_type='application/json')
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)
        self.assertIn("error", response.data)

    def test_me_authenticated(self):
        response = self.client.get('/api/auth/me', **self.auth_headers)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["email"], "sankaran@gmail.com")
        self.assertEqual(response.data["fullName"], "Sankaran S")

    def test_me_unauthenticated(self):
        response = self.client.get('/api/auth/me')
        self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)

    def test_dashboard_stats(self):
        # Mock study logs collection (e.g. 1 entry logged for today)
        self.collections['study_logs'] = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.sort.return_value = [
            {"userId": self.user_id, "date": datetime.date.today().strftime("%Y-%m-%d"), "duration_minutes": 30}
        ]
        self.collections['study_logs'].find.return_value = mock_cursor

        # Mock chats collection
        self.collections['chats'] = MagicMock()
        mock_chats_cursor = MagicMock()
        mock_chats_cursor.sort.return_value = []
        self.collections['chats'].find.return_value = mock_chats_cursor

        # Mock study goals
        self.collections['study_goals'] = MagicMock()
        self.collections['study_goals'].find_one.return_value = {"userId": self.user_id, "target_weekly_hours": 8}

        response = self.client.get('/api/dashboard/stats', **self.auth_headers)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["fullName"], "Sankaran S")
        self.assertEqual(response.data["goals"]["targetHours"], 8)
        self.assertEqual(response.data["streak"]["currentStreak"], 1)

    def test_log_study(self):
        # Mock study logs collection
        self.collections['study_logs'] = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.sort.return_value = []
        self.collections['study_logs'].find.return_value = mock_cursor

        payload = {
            "duration_minutes": 25,
            "activity_type": "chat"
        }
        response = self.client.post('/api/dashboard/log-study', payload, content_type='application/json', **self.auth_headers)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("Successfully logged", response.data["message"])
        self.assertEqual(response.data["todayMinutes"], 25)

    def test_update_goals(self):
        self.collections['study_goals'] = MagicMock()

        payload = {
            "target_weekly_hours": 10
        }
        response = self.client.post('/api/dashboard/goals', payload, content_type='application/json', **self.auth_headers)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["targetWeeklyHours"], 10)

    def test_chat_session_create(self):
        self.collections['chats'] = MagicMock()
        self.collections['chats'].insert_one.return_value = MagicMock(inserted_id=ObjectId())

        response = self.client.post('/api/chat/session/create', **self.auth_headers)
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertIn("sessionId", response.data)

    def test_query_tutor_rag(self):
        session_id = str(uuid.uuid4())

        # Mock chat session
        mock_chats = MagicMock()
        mock_chats.find_one.return_value = {
            "_id": ObjectId(),
            "sessionId": session_id,
            "userId": str(self.user_id),
            "chats": []
        }
        self.collections['chats'] = mock_chats

        # Prevent MagicMock cache hits in query_cache
        mock_cache = MagicMock()
        mock_cache.find_one.return_value = None
        self.collections['query_cache'] = mock_cache

        # Save original RAG service and Groq references
        import tutor_app.utils.rag_service as rs
        import langchain_groq

        orig_classify = rs.classify_query_intent
        orig_fetch = rs.fetch_diagram_from_qdrant
        orig_get_emb = rs.get_embeddings
        orig_chat_groq = langchain_groq.ChatGroq

        # Clean monkey-patches to prevent RecursionError
        rs.classify_query_intent = lambda query, last: {
            "is_academic": True,
            "is_followup": False,
            "is_image": True,
            "image_name": "photosynthesis process",
            "subject": ["Science", "Biology"],
        }

        rs.fetch_diagram_from_qdrant = lambda image_name, grade, *args, **kwargs: {
            "imageUrl": "https://cloudinary.com/diagram.png",
            "topic": "Photosynthesis Diagram",
            "caption": "Diagram showing photosynthesis process",
            "score": 0.91,
        }

        class FakeEmbeddings:
            def embed_query(self, text):
                return [0.1] * 384

        rs.get_embeddings = lambda: FakeEmbeddings()

        # Mock Qdrant textbook retrieval client
        mock_qdrant = MagicMock()
        mock_point = MagicMock()
        mock_point.score = 0.85
        mock_point.payload = {
            "page_content": "Photosynthesis is the process by which green plants make food.",
            "metadata": {"subject": "Science", "chapter": "Photosynthesis", "page_number": "12"}
        }
        mock_qdrant.search.return_value = [mock_point]
        mock_res = MagicMock()
        mock_res.points = [mock_point]
        mock_qdrant.query_points.return_value = mock_res
        rs.get_qdrant_client = lambda: mock_qdrant

        # Clean mock for Groq LLM class
        mock_groq_res = MagicMock()
        mock_groq_res.content = "Photosynthesis is indeed a plant process."

        class FakeChatGroq:
            def __init__(self, *args, **kwargs):
                pass
            def invoke(self, messages):
                return mock_groq_res

        langchain_groq.ChatGroq = FakeChatGroq

        try:
            response = self.client.post(
                f'/api/chat/session/{session_id}/query',
                {"query": "explain photosynthesis"},
                content_type='application/json',
                **self.auth_headers
            )
            self.assertEqual(response.status_code, status.HTTP_200_OK)
            self.assertIn("response", response.data)
            self.assertEqual(response.data["response"], "Photosynthesis is indeed a plant process.")
            self.assertEqual(response.data["sources"][0]["subject"], "Science")
            self.assertIsNotNone(response.data["diagram"])
            self.assertEqual(response.data["diagram"]["topic"], "Photosynthesis Diagram")
        finally:
            # Restore references
            rs.classify_query_intent = orig_classify
            rs.fetch_diagram_from_qdrant = orig_fetch
            rs.get_embeddings = orig_get_emb
            langchain_groq.ChatGroq = orig_chat_groq

    def test_forgot_password_success(self):
        payload = {
            "email": "sankaran@gmail.com"
        }
        response = self.client.post('/api/auth/forgot-password', payload, content_type='application/json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("reset code has been sent", response.data["message"])
        self.assertTrue(self.collections['users'].update_one.called)

    def test_reset_password_success(self):
        # Update user mock to contain valid OTP
        self.mock_user["resetOtp"] = "123456"
        self.mock_user["resetOtpExpiresAt"] = (datetime.datetime.utcnow() + datetime.timedelta(minutes=10)).isoformat()

        payload = {
            "email": "sankaran@gmail.com",
            "code": "123456",
            "new_password": "NewSecurePassword555"
        }
        response = self.client.post('/api/auth/reset-password', payload, content_type='application/json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("reset successfully", response.data["message"])
        self.assertTrue(self.collections['users'].update_one.called)

    def test_reset_password_invalid_otp(self):
        self.mock_user["resetOtp"] = "123456"
        self.mock_user["resetOtpExpiresAt"] = (datetime.datetime.utcnow() + datetime.timedelta(minutes=10)).isoformat()

        payload = {
            "email": "sankaran@gmail.com",
            "code": "999999", # wrong code
            "new_password": "NewSecurePassword555"
        }
        response = self.client.post('/api/auth/reset-password', payload, content_type='application/json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("Invalid email or verification code", response.data["error"])

    def test_update_profile_success(self):
        payload = {
            "fullName": "Sankaran S Updated",
            "grade": "9"
        }
        # Mock update_one and find_one on reload
        self.collections['users'].update_one = MagicMock()
        self.collections['users'].find_one.return_value = {
            "_id": ObjectId(self.user_id),
            "fullName": "Sankaran S Updated",
            "email": "sankaran@gmail.com",
            "password": "hashed_password",
            "role": "student",
            "studentClass": {"grade": "8", "section": "A", "rollNumber": "10"},
            "isActive": True
        }

        response = self.client.put('/api/auth/me', payload, content_type='application/json', **self.auth_headers)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["fullName"], "Sankaran S Updated")
        self.assertEqual(response.data["studentClass"]["grade"], "8")
        self.assertTrue(self.collections['users'].update_one.called)

    def test_change_password_success(self):
        payload = {
            "current_password": "securepass",
            "new_password": "NewSecurePassword777"
        }
        self.collections['users'].update_one = MagicMock()
        
        response = self.client.post('/api/auth/change-password', payload, content_type='application/json', **self.auth_headers)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("Password changed successfully", response.data["message"])
        self.assertTrue(self.collections['users'].update_one.called)

    def test_change_password_wrong_current(self):
        payload = {
            "current_password": "wrongpassword",
            "new_password": "NewSecurePassword777"
        }
        response = self.client.post('/api/auth/change-password', payload, content_type='application/json', **self.auth_headers)
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("Incorrect current password", response.data["error"])

    def test_get_sample_questions_success(self):
        self.collections['gradequestions'] = MagicMock()
        self.collections['gradequestions'].find_one.return_value = {
            "_id": ObjectId(),
            "grade": "8",
            "questions": [
                {"name": "Science", "questions": "Define evaporation?", "_id": ObjectId()},
                {"name": "Maths", "questions": "Calculate 5+5", "_id": ObjectId()}
            ]
        }

        response = self.client.get('/api/chat/sample-questions', **self.auth_headers)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data), 2)
        self.assertEqual(response.data[0]["subject"], "Science")
        self.assertEqual(response.data[0]["question"], "Define evaporation?")
        self.assertEqual(response.data[1]["subject"], "Maths")
        self.assertEqual(response.data[1]["question"], "Calculate 5+5")

    def test_get_sample_questions_unauthenticated(self):
        response = self.client.get('/api/chat/sample-questions')
        self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)

    def test_list_books_success(self):
        book_id = ObjectId()
        self.collections['book_chapters'] = MagicMock()
        self.collections['book_chapters'].find.return_value = [
            {
                "_id": book_id,
                "schoolId": "school_123",
                "board": "CBSE",
                "class": "8",
                "subject": "Tamil",
                "upload_status": "success",
                "chapters": [
                    {"chapter_no": "1", "chapter_name": "Chapter 1 Name"}
                ],
                "total_chapters": 1
            }
        ]

        self.collections['chapter_summaries'] = MagicMock()
        self.collections['chapter_summaries'].find.return_value = [
            {
                "_id": ObjectId(),
                "chapter_ref": book_id,
                "chapter_no": "1",
                "chapter_name": "Chapter 1 Name",
                "summary_text": "Sample summary text content"
            }
        ]

        response = self.client.get('/api/content/books', **self.auth_headers)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data), 1)
        self.assertEqual(response.data[0]["subject"], "Tamil")
        self.assertEqual(response.data[0]["chapters"][0]["chapterNo"], "1")
        self.assertEqual(response.data[0]["chapters"][0]["summary"], "Sample summary text content")

    def test_list_books_unauthenticated(self):
        response = self.client.get('/api/content/books')
        self.assertEqual(response.status_code, status.HTTP_403_FORBIDDEN)

