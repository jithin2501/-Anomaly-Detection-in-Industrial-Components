from services import history_service


def handle_history(db, limit: int = 50):
    return history_service.list_recent(db, limit)
