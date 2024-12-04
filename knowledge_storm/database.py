from dotenv import dotenv_values
import mariadb
import os

from alembic.config import Config
from alembic import command

from sqlalchemy import create_engine, desc, inspect, Column, Integer, String, Time, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, mapped_column


Base = declarative_base()

class User(Base):
    __tablename__ = 'user'
    id = Column(Integer, primary_key=True)
    first_name = Column(String(50))
    last_name = Column(String(100))

class Topic(Base):
    __tablename__ = 'topic'
    id = Column(Integer, primary_key=True)
    topic = Column(String(100))
    start_date = Column(Time)
    end_date = Column(Time)
    status = Column(String(5000))
    pdf_documents = relationship("PDFDocument", back_populates="topic")
    csv_documents = relationship("CSVDocument", back_populates="topic")
    md_documents = relationship("MDDocument", back_populates="topic")
    json_documents = relationship("JSONDocument", back_populates="topic")


class PDFDocument(Base):
    __tablename__ = 'pdf_documents'
    id = Column(Integer, primary_key=True)
    object_link = Column(String(1024))
    topic_id = mapped_column(ForeignKey("topic.id"))
    topic = relationship("Topic", back_populates="pdf_documents")

class CSVDocument(Base):
    __tablename__ = 'csv_documents'
    id = Column(Integer, primary_key=True)
    object_link = Column(String(1024))
    topic_id = mapped_column(ForeignKey("topic.id"))
    topic = relationship("Topic", back_populates="csv_documents")

class MDDocument(Base):
    __tablename__ = 'md_documents'
    id = Column(Integer, primary_key=True)
    object_link = Column(String(1024))
    topic_id = mapped_column(ForeignKey("topic.id"))
    topic = relationship("Topic", back_populates="md_documents")

class JSONDocument(Base):
    __tablename__ = 'json_documents'
    id = Column(Integer, primary_key=True)
    object_link = Column(String(1024))
    topic_id = mapped_column(ForeignKey("topic.id"))
    topic = relationship("Topic", back_populates="json_documents")

class DataBase:
    def __init__(self, mariadbHost, mariadbPassword, mariadbDb, port = 3306):
        print(f"Using port: {port}")
        print(f"Connecting to database at {mariadbHost}:{port}, database: {mariadbDb}")
        self.engine = create_engine(f'mariadb+mariadbconnector://root:{mariadbPassword}@{mariadbHost}:{port}/{mariadbDb}')
        Base.metadata.create_all(self.engine, checkfirst=True)
        self.Session = sessionmaker(bind=self.engine)

    def create_session(self):
        return self.Session()
    
    def check_tables(self):
        inspector = inspect(self.engine)

        tables = inspector.get_table_names()
        print(tables)
    
    def check_user(self):
        for column in User.__table__.columns:
            print(f"colonne{column.name}")
    
    def update_status_topic(self, topic, status):
        session = self.create_session()
        latest_topic = (
            session.query(Topic)
            .filter_by(topic=topic)
            .order_by(desc(Topic.start_date))
            .first()
        )
        if latest_topic:
            latest_topic.status = status
            session.commit()
            print(latest_topic.status)
        else:
            print("No matching topic found.")
        session.close()