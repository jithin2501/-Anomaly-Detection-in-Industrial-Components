from models.db_models import Inspection


def save(db, filename, label, score, threshold):
    row = Inspection(filename=filename, label=label, score=float(score), threshold=float(threshold))
    db.add(row)
    db.commit()
    return row


def list_recent(db, limit=50):
    return db.query(Inspection).order_by(Inspection.id.desc()).limit(limit).all()
