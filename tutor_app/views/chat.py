import uuid
import datetime
from bson import ObjectId
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from tutor_app.utils.db_client import get_collection
from tutor_app.utils.rag_service import execute_rag_tutor_query

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def list_sessions_view(request):
    """
    GET /api/chat/sessions
    Lists all active (non-archived) chat sessions for the authenticated student.
    """
    try:
        user_id = request.user.id
        chats_col = get_collection('chats')
        if chats_col is None:
            return Response(
                {"error": "Database service is currently unavailable."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        # Retrieve and sort by pinned first, then by updatedAt descending
        search_query = request.GET.get('q', '').strip()
        find_query = {
            "$and": [
                {"$or": [{"userId": str(user_id)}, {"userId": ObjectId(user_id)}]},
                {"archived": {"$ne": True}}
            ]
        }
        if search_query:
            find_query["$and"].append({
                "title": {"$regex": search_query, "$options": "i"}
            })

        sessions = list(
            chats_col.find(find_query).sort([("pinned", -1), ("updatedAt", -1)])
        )

        serialized = []
        for sess in sessions:
            messages = sess.get("chats", [])
            last_message_preview = ""
            if messages:
                last_message_preview = messages[-1].get("query", "")[:50]

            serialized.append({
                "id": str(sess["_id"]),
                "sessionId": sess.get("sessionId"),
                "title": sess.get("title", "New Chat"),
                "grade": sess.get("grade", "8"),
                "subject": sess.get("subject", "Science"),
                "pinned": sess.get("pinned", False),
                "lastMessagePreview": last_message_preview,
                "updatedAt": sess.get("updatedAt")
            })

        return Response(serialized, status=status.HTTP_200_OK)

    except Exception as e:
        return Response(
            {"error": f"Failed to list chat sessions: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def create_session_view(request):
    """
    POST /api/chat/session/create
    Creates a new chat session document in the database.
    """
    try:
        user_id = request.user.id
        chats_col = get_collection('chats')
        if chats_col is None:
            return Response(
                {"error": "Database service is currently unavailable."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        title = request.data.get('title')
        title_str = str(title).strip() if title else "New Chat"
        subject = str(request.data.get('subject', 'Science')).strip()

        session_id = str(uuid.uuid4())
        now_str = datetime.datetime.utcnow().isoformat() + "Z"
        grade = request.user.grade or '8'

        new_session = {
            "sessionId": session_id,
            "userId": str(user_id),
            "title": title_str,
            "grade": grade,
            "subject": subject,
            "chats": [],
            "pinned": False,
            "archived": False,
            "createdAt": now_str,
            "updatedAt": now_str
        }

        result = chats_col.insert_one(new_session)

        return Response({
            "id": str(result.inserted_id),
            "sessionId": session_id,
            "title": title_str,
            "pinned": False,
            "chats": [],
            "createdAt": now_str,
            "updatedAt": now_str
        }, status=status.HTTP_201_CREATED)

    except Exception as e:
        return Response(
            {"error": f"Failed to create chat session: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET', 'DELETE'])
@permission_classes([IsAuthenticated])
def session_detail_view(request, session_id):
    """
    GET /api/chat/session/<session_id>
    Retrieves full details and message history of a specific session.

    DELETE /api/chat/session/<session_id>
    Archives a chat session (sets archived flag to True).
    """
    try:
        user_id = request.user.id
        chats_col = get_collection('chats')
        if chats_col is None:
            return Response(
                {"error": "Database service is currently unavailable."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        # Look up session by sessionId uuid string or _id ObjectId string
        query = {
            "$or": [{"sessionId": session_id}],
            "$or": [{"userId": str(user_id)}, {"userId": ObjectId(user_id)}]
        }
        
        # Add support for _id lookup as well
        if len(session_id) == 24:
            try:
                query["$or"].append({"_id": ObjectId(session_id)})
            except Exception:
                pass
                
        # Main query check
        session = chats_col.find_one({
            "$and": [
                {"$or": [{"sessionId": session_id}, {"_id": ObjectId(session_id) if len(session_id) == 24 else None}]},
                {"$or": [{"userId": str(user_id)}, {"userId": ObjectId(user_id)}]}
            ]
        })

        if not session:
            return Response(
                {"error": "Chat session not found or unauthorized access."},
                status=status.HTTP_404_NOT_FOUND
            )

        if request.method == 'GET':
            return Response({
                "id": str(session["_id"]),
                "sessionId": session.get("sessionId"),
                "title": session.get("title", "New Chat"),
                "pinned": session.get("pinned", False),
                "chats": session.get("chats", []),
                "createdAt": session.get("createdAt"),
                "updatedAt": session.get("updatedAt")
            }, status=status.HTTP_200_OK)

        elif request.method == 'DELETE':
            # Soft delete (archive)
            chats_col.update_one(
                {"_id": session["_id"]},
                {"$set": {
                    "archived": True,
                    "updatedAt": datetime.datetime.utcnow().isoformat()
                }}
            )
            return Response(
                {"message": "Chat session successfully archived."},
                status=status.HTTP_200_OK
            )

    except Exception as e:
        return Response(
            {"error": f"An error occurred executing session operation: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def query_tutor_view(request, session_id):
    """
    POST /api/chat/session/<session_id>/query
    Processes a student query using the RAG model and appends messages to history.
    Expects JSON: { "query": str }
    """
    try:
        user_id = request.user.id
        query_text = request.data.get("query")

        if not query_text or not str(query_text).strip():
            return Response(
                {"error": "Query string is required."},
                status=status.HTTP_400_BAD_REQUEST
            )

        if len(str(query_text)) > 5000:
            return Response(
                {"error": "Query string exceeds maximum allowed length of 5000 characters."},
                status=status.HTTP_400_BAD_REQUEST
            )

        chats_col = get_collection('chats')
        if chats_col is None:
            return Response(
                {"error": "Database service is currently unavailable."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        # Look up session and confirm authorization
        session = chats_col.find_one({
            "$and": [
                {"$or": [{"sessionId": session_id}, {"_id": ObjectId(session_id) if len(session_id) == 24 else None}]},
                {"$or": [{"userId": str(user_id)}, {"userId": ObjectId(user_id)}]}
            ]
        })

        if not session:
            return Response(
                {"error": "Chat session not found or unauthorized access."},
                status=status.HTTP_404_NOT_FOUND
            )

        # Execute RAG query (passing active grade and existing chat history)
        history = session.get("chats", [])
        debug = request.data.get("debug", False)
        cache = request.data.get("cache", True)

        result = execute_rag_tutor_query(
            query_text=str(query_text).strip(),
            grade=request.user.grade,
            chat_history=history,
            debug=debug,
            cache=cache,
        )

        now_str = datetime.datetime.utcnow().isoformat() + "Z"

        # Construct message object for DB (exclude debug logs from database bloat)
        db_message = {
            "query":     str(query_text).strip(),
            "response":  result["response"],
            "sources":   result["sources"],
            "diagram":   result["diagram"],
            "subject":   result.get("subject", ["Science"]),
            "timestamp": now_str,
        }

        # Automatically update session title if it is currently a default title
        current_title = session.get("title", "")
        update_set = {
            "updatedAt": now_str,
            "subject": (result.get("subject") or ["Science"])[0],
        }
        if current_title in ["New Learning Session", "New Chat", "New Chat Session"]:
            truncated_title = str(query_text).strip()
            if len(truncated_title) > 40:
                truncated_title = truncated_title[:37] + "..."
            update_set["title"] = truncated_title

        # Append messages to session; also update session-level subject from LLM
        chats_col.update_one(
            {"_id": session["_id"]},
            {
                "$push": {"chats": db_message},
                "$set": update_set,
            },
        )

        # Include debug information in response if requested
        response_payload = dict(db_message)
        if "debug_info" in result:
            response_payload["debug_info"] = result["debug_info"]

        return Response(response_payload, status=status.HTTP_200_OK)


    except Exception as e:
        return Response(
            {"error": f"An error occurred executing query: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def toggle_pin_session_view(request, session_id):
    """
    POST /api/chat/session/<session_id>/pin
    Toggles the pinned status of a chat session.
    """
    try:
        user_id = request.user.id
        chats_col = get_collection('chats')
        if chats_col is None:
            return Response(
                {"error": "Database service is currently unavailable."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        # Look up session and confirm authorization
        session = chats_col.find_one({
            "$and": [
                {"$or": [{"sessionId": session_id}, {"_id": ObjectId(session_id) if len(session_id) == 24 else None}]},
                {"$or": [{"userId": str(user_id)}, {"userId": ObjectId(user_id)}]}
            ]
        })

        if not session:
            return Response(
                {"error": "Chat session not found or unauthorized access."},
                status=status.HTTP_404_NOT_FOUND
            )

        new_pinned_state = not session.get("pinned", False)

        chats_col.update_one(
            {"_id": session["_id"]},
            {"$set": {
                "pinned": new_pinned_state,
                "updatedAt": datetime.datetime.utcnow().isoformat()
            }}
        )

        return Response({
            "message": f"Session successfully {'pinned' if new_pinned_state else 'unpinned'}.",
            "pinned": new_pinned_state
        }, status=status.HTTP_200_OK)

    except Exception as e:
        return Response(
            {"error": f"An error occurred pinning session: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def sample_questions_view(request):
    """
    GET /api/chat/sample-questions
    Retrieves the list of syllabus sample questions scoped to the authenticated student's grade level.
    """
    try:
        user_grade = request.user.grade
        if not user_grade:
            return Response(
                {"error": "Student grade level is not configured in profile."},
                status=status.HTTP_400_BAD_REQUEST
            )

        gradequestions_col = get_collection('gradequestions')
        if gradequestions_col is None:
            return Response(
                {"error": "Database service is currently unavailable."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        doc = gradequestions_col.find_one({"grade": str(user_grade)})
        if not doc:
            return Response([], status=status.HTTP_200_OK)

        raw_questions = doc.get("questions", [])
        serialized = []
        for q in raw_questions:
            serialized.append({
                "id": str(q.get("_id")) if q.get("_id") else "",
                "subject": q.get("name", "General"),
                "question": q.get("questions", "")
            })

        return Response(serialized, status=status.HTTP_200_OK)

    except Exception as e:
        return Response(
            {"error": f"An error occurred retrieving sample questions: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
def debug_hf_view(request):
    import traceback
    import os
    from tutor_app.utils.rag_service import HuggingFaceInferenceEmbeddings
    
    debug_info = {}
    debug_info["HF_TOKEN_exists"] = bool(os.getenv("HF_TOKEN"))
    debug_info["HF_TOKEN_len"] = len(os.getenv("HF_TOKEN", ""))
    debug_info["EMBEDDING_MODEL"] = os.getenv("EMBEDDING_MODEL")
    
    try:
        emb = HuggingFaceInferenceEmbeddings()
        debug_info["init_success"] = True
        debug_info["api_token_exists"] = bool(emb.api_token)
        debug_info["api_token_len"] = len(emb.api_token) if emb.api_token else 0
        debug_info["model_name"] = emb.model_name
        
        # Try a quick feature extraction
        res = emb.client.feature_extraction(text="hello", model=emb.model_name)
        debug_info["extraction_success"] = True
        debug_info["extraction_result_type"] = str(type(res))
    except Exception as e:
        debug_info["init_success"] = False
        debug_info["error"] = str(e)
        debug_info["traceback"] = traceback.format_exc()
        
    return Response(debug_info, status=200)


