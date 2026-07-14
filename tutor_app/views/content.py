from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from tutor_app.utils.db_client import get_collection

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def list_books_view(request):
    """
    GET /api/content/books
    Retrieves uploaded books, chapters, and chapter summaries matching the student's grade level.
    """
    try:
        grade = getattr(request.user, 'grade', None)
        if not grade:
            # Fallback or error
            return Response(
                {"error": "User does not have a grade associated with their profile."},
                status=status.HTTP_400_BAD_REQUEST
            )

        book_chapters_col = get_collection('book_chapters')
        chapter_summaries_col = get_collection('chapter_summaries')

        if book_chapters_col is None or chapter_summaries_col is None:
            return Response(
                {"error": "Database service is currently unavailable."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        # Retrieve successful books matching student's grade
        # We query by converting grade to string just in case
        books = list(book_chapters_col.find({
            "class": str(grade),
            "upload_status": "success"
        }))

        if not books:
            return Response([], status=status.HTTP_200_OK)

        # Collect book ObjectIds to fetch matching summaries
        book_ids = [b["_id"] for b in books]
        summaries = list(chapter_summaries_col.find({
            "chapter_ref": {"$in": book_ids}
        }))

        # Create a lookup map of (book_id_str, chapter_no_str) -> summary_text
        summary_lookup = {}
        for s in summaries:
            ref_id = str(s.get("chapter_ref"))
            ch_no = str(s.get("chapter_no", "")).strip()
            summary_text = s.get("summary_text", "")
            if ref_id and ch_no:
                summary_lookup[(ref_id, ch_no)] = summary_text

        # Serialize and merge data
        serialized_books = []
        for b in books:
            book_id_str = str(b["_id"])
            chapters = []
            for ch in b.get("chapters", []):
                ch_no = str(ch.get("chapter_no", "")).strip()
                ch_name = ch.get("chapter_name", "")
                summary_text = summary_lookup.get((book_id_str, ch_no), "")
                
                chapters.append({
                    "chapterNo": ch_no,
                    "chapterName": ch_name,
                    "summary": summary_text
                })
            
            serialized_books.append({
                "bookId": book_id_str,
                "subject": b.get("subject", ""),
                "board": b.get("board", ""),
                "class": b.get("class", ""),
                "totalChapters": b.get("total_chapters", len(chapters)),
                "chapters": chapters
            })

        return Response(serialized_books, status=status.HTTP_200_OK)

    except Exception as e:
        return Response(
            {"error": f"Failed to retrieve books and summaries: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
