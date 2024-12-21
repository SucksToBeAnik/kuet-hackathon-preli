from sqlmodel import SQLModel, Session, create_engine
from config.settings import settings

engine = create_engine(settings.database_url)


def get_session():
    try:
        session = Session(engine)
        yield session
    except Exception as e:
        print(e)
        raise e
