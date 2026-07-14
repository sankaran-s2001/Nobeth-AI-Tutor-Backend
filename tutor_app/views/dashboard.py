import datetime
from bson import ObjectId
from collections import defaultdict
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from tutor_app.utils.db_client import get_collection

def get_streak_info(user_id):
    """
    Computes the student's consecutive study streak (in days)
    and 7-day daily study log calendar list.
    """
    try:
        logs_col = get_collection('study_logs')
        if logs_col is None:
            return 0, [], {}

        # Fetch all logs for this user, sorted descending by date string (YYYY-MM-DD)
        logs = list(logs_col.find({"userId": ObjectId(user_id)}).sort("date", -1))

        # Accumulate minutes per date string
        daily_durations = defaultdict(int)
        for log in logs:
            date_str = log.get('date')
            if date_str:
                daily_durations[date_str] += log.get('duration_minutes', 0)

        today = datetime.date.today()
        today_str = today.strftime("%Y-%m-%d")
        yesterday_str = (today - datetime.timedelta(days=1)).strftime("%Y-%m-%d")

        streak = 0
        check_date = today

        # Determine starting day for consecutive streak calculation
        if daily_durations[today_str] > 0:
            # Started studying today
            while True:
                d_str = check_date.strftime("%Y-%m-%d")
                if daily_durations[d_str] > 0:
                    streak += 1
                    check_date -= datetime.timedelta(days=1)
                else:
                    break
        elif daily_durations[yesterday_str] > 0:
            # Active yesterday but not logged today yet
            check_date = today - datetime.timedelta(days=1)
            while True:
                d_str = check_date.strftime("%Y-%m-%d")
                if daily_durations[d_str] > 0:
                    streak += 1
                    check_date -= datetime.timedelta(days=1)
                else:
                    break
        else:
            # Streak broken
            streak = 0

        # Construct current calendar week (Monday to Sunday) daily study log calendar
        # today.weekday() is 0 for Monday, 1 for Tuesday, ..., 6 for Sunday
        monday = today - datetime.timedelta(days=today.weekday())
        weekly_history = []
        for i in range(7):
            d = monday + datetime.timedelta(days=i)
            d_str = d.strftime("%Y-%m-%d")
            # If the day is in the future, don't show any study logs
            duration = daily_durations[d_str] if d <= today else 0
            weekly_history.append({
                "date": d_str,
                "dayName": d.strftime("%a"),
                "duration": duration,
                "studied": duration > 0
            })

        return streak, weekly_history, daily_durations

    except Exception:
        # Graceful fallback in case of database or parsing exceptions
        return 0, [], {}


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def dashboard_stats_view(request):
    """
    GET /api/dashboard/stats
    Retrieves all dashboard statistics and widget metrics for the authenticated student.
    """
    try:
        user_id = request.user.id
        
        # 1. Calculate streak and 7-day history
        streak, weekly_history, daily_durations = get_streak_info(user_id)

        # 2. Get today's total study minutes
        today_str = datetime.date.today().strftime("%Y-%m-%d")
        today_minutes = daily_durations.get(today_str, 0)

        # 3. Sum weekly study duration in minutes
        weekly_minutes = sum(day['duration'] for day in weekly_history)
        weekly_hours = weekly_minutes // 60
        weekly_mins = weekly_minutes % 60
        study_time_display = f"{weekly_hours}h {weekly_mins}m" if weekly_hours > 0 else f"{weekly_mins}m"

        # 4. Fetch study target goals
        goals_col = get_collection('study_goals')
        target_weekly_hours = 6  # Default goal is 6h as shown in mockup UI
        if goals_col is not None:
            goal_doc = goals_col.find_one({"userId": ObjectId(user_id)})
            if goal_doc:
                target_weekly_hours = goal_doc.get("target_weekly_hours", 6)

        target_weekly_minutes = target_weekly_hours * 60
        goal_percentage = 0
        if target_weekly_minutes > 0:
            goal_percentage = min(100, int((weekly_minutes / target_weekly_minutes) * 100))

        # 5. Retrieve counts of Chats and Questions asked in the last 7 days
        chats_col = get_collection('chats')
        total_chats = 0
        questions_asked = 0
        last_chat_info = None

        if chats_col is not None:
            # Query active sessions (matches both string or ObjectId userId formats)
            active_sessions = list(
                chats_col.find({
                    "$or": [{"userId": str(user_id)}, {"userId": ObjectId(user_id)}],
                    "archived": {"$ne": True}
                }).sort("updatedAt", -1)
            )
            
            total_chats = len(active_sessions)
            seven_days_ago = datetime.datetime.utcnow() - datetime.timedelta(days=7)

            for sess in active_sessions:
                messages = sess.get("chats", [])
                for msg in messages:
                    ts_str = msg.get("timestamp")
                    if ts_str:
                        try:
                            ts = datetime.datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                            ts_naive = ts.astimezone(datetime.timezone.utc).replace(tzinfo=None)
                            if ts_naive >= seven_days_ago:
                                questions_asked += 1
                        except Exception:
                            questions_asked += 1  # Fallback to count in case of parsing format anomalies
                    else:
                        questions_asked += 1

            # Populate the "Continue Learning" widget using the single most recent chat
            if active_sessions:
                most_recent = active_sessions[0]
                messages = most_recent.get("chats", [])
                if messages:
                    last_msg = messages[-1]
                    subject_name = "Science"
                    if last_msg.get("sources"):
                        subject_name = last_msg.get("sources")[0].get("subject", "Science")

                    last_chat_info = {
                        "sessionId": most_recent.get("sessionId"),
                        "query": last_msg.get("query"),
                        "subject": subject_name,
                        "timestamp": last_msg.get("timestamp")
                    }

        # Structure response strictly matching UI widgets
        return Response({
            "fullName": request.user.name,
            "selectedClass": f"Class {request.user.grade}",
            "stats": {
                "totalChats": total_chats,
                "studyTime": study_time_display,
                "questionsAsked": questions_asked,
                "accuracy": "96%" # Static metric helper
            },
            "continueLearning": last_chat_info,
            "streak": {
                "currentStreak": streak,
                "weeklyHistory": weekly_history
            },
            "goals": {
                "targetHours": target_weekly_hours,
                "completedMinutes": weekly_minutes,
                "percentage": goal_percentage
            }
        }, status=status.HTTP_200_OK)

    except Exception as e:
        return Response(
            {"error": f"An error occurred loading dashboard statistics: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def log_study_view(request):
    """
    POST /api/dashboard/log-study
    Logs study duration in minutes for the authenticated student.
    Expects JSON: { "duration_minutes": int, "activity_type": str }
    """
    try:
        user_id = request.user.id
        duration = request.data.get('duration_minutes')
        activity = request.data.get('activity_type', 'chat')

        if duration is None:
            return Response(
                {"error": "duration_minutes parameter is required."},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            duration = int(duration)
            if duration <= 0:
                raise ValueError()
        except ValueError:
            return Response(
                {"error": "duration_minutes must be a positive integer."},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Offload DB write to Celery Task Queue (runs synchronously for immediate updates)
        from tutor_app.tasks import log_study_time_async
        try:
            log_study_time_async(str(user_id), duration, activity)
        except Exception as e:
            logger.error(f"Failed synchronous task execution, falling back to celery delay: {e}")
            log_study_time_async.delay(str(user_id), duration, activity)

        # Retrieve current stats & manually add the new duration for instant response (race condition safety)
        streak, _, daily_durations = get_streak_info(user_id)
        today_str = datetime.date.today().strftime("%Y-%m-%d")
        today_total = daily_durations.get(today_str, 0) + duration

        if daily_durations.get(today_str, 0) == 0:
            yesterday_str = (datetime.date.today() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
            if daily_durations.get(yesterday_str, 0) > 0 or streak == 0:
                streak += 1

        return Response({
            "message": f"Successfully logged {duration} minutes of study.",
            "todayMinutes": today_total,
            "streak": streak
        }, status=status.HTTP_200_OK)

    except Exception as e:
        return Response(
            {"error": f"An error occurred logging study time: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def update_goal_view(request):
    """
    POST /api/dashboard/goals
    Updates the student's weekly study goal target hours.
    Expects JSON: { "target_weekly_hours": int }
    """
    try:
        user_id = request.user.id
        target = request.data.get('target_weekly_hours')

        if target is None:
            return Response(
                {"error": "target_weekly_hours parameter is required."},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            target = int(target)
            if target <= 0 or target > 168:
                raise ValueError()
        except ValueError:
            return Response(
                {"error": "target_weekly_hours must be a positive integer and cannot exceed 168 hours."},
                status=status.HTTP_400_BAD_REQUEST
            )

        goals_col = get_collection('study_goals')
        if goals_col is None:
            return Response(
                {"error": "Database service is currently offline."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        # Set/upsert target weekly hours
        goals_col.update_one(
            {"userId": ObjectId(user_id)},
            {
                "$set": {
                    "target_weekly_hours": target,
                    "updated_at": datetime.datetime.utcnow().isoformat()
                }
            },
            upsert=True
        )

        return Response({
            "message": "Weekly target hours successfully updated.",
            "targetWeeklyHours": target
        }, status=status.HTTP_200_OK)

    except Exception as e:
        return Response(
            {"error": f"An error occurred saving goals configuration: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
