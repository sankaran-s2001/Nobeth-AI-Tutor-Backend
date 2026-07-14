from django.contrib import admin
from django.urls import path
from tutor_app.views.auth import login_view, me_view, logout_view, forgot_password_view, reset_password_view, change_password_view
from tutor_app.views.dashboard import dashboard_stats_view, log_study_view, update_goal_view
from tutor_app.views.chat import (
    list_sessions_view, create_session_view, session_detail_view,
    query_tutor_view, toggle_pin_session_view, sample_questions_view
)
from tutor_app.views.content import list_books_view

urlpatterns = [
    path('admin/', admin.site.urls),
    
    # ── Authentication Endpoints ──
    path('api/auth/login', login_view, name='login'),
    path('api/auth/me', me_view, name='me'),
    path('api/auth/logout', logout_view, name='logout'),
    path('api/auth/forgot-password', forgot_password_view, name='forgot_password'),
    path('api/auth/reset-password', reset_password_view, name='reset_password'),
    path('api/auth/change-password', change_password_view, name='change_password'),
    
    # ── Dashboard Endpoints ──
    path('api/dashboard/stats', dashboard_stats_view, name='dashboard_stats'),
    path('api/dashboard/log-study', log_study_view, name='log_study'),
    path('api/dashboard/goals', update_goal_view, name='update_goal'),
    
    # ── Chat & RAG Endpoints ──
    path('api/chat/sessions', list_sessions_view, name='list_sessions'),
    path('api/chat/sample-questions', sample_questions_view, name='sample_questions'),
    path('api/chat/session/create', create_session_view, name='create_session'),
    path('api/chat/session/<str:session_id>', session_detail_view, name='session_detail'),
    path('api/chat/session/<str:session_id>/query', query_tutor_view, name='query_tutor'),
    path('api/chat/session/<str:session_id>/pin', toggle_pin_session_view, name='toggle_pin_session'),
    
    # ── Content Management Endpoints ──
    path('api/content/books', list_books_view, name='list_books'),
]

