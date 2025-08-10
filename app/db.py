from __future__ import annotations
from sqlalchemy import create_engine, Column, Integer, String, Date, DateTime, ForeignKey, UniqueConstraint, text
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime, date
import os

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'attendance.sqlite3')
ENGINE = create_engine(f'sqlite:///{os.path.abspath(DB_PATH)}', future=True, echo=False)
SessionLocal = sessionmaker(bind=ENGINE, autoflush=False, autocommit=False, future=True)
Base = declarative_base()


class Student(Base):
    __tablename__ = 'students'
    id = Column(Integer, primary_key=True)
    code = Column(String, nullable=False, unique=True)  # código único del estudiante
    name = Column(String, nullable=False)
    grade = Column(String, nullable=True)
    section = Column(String, nullable=True)
    gender = Column(String, nullable=True)
    registration_date = Column(Date, nullable=True)
    photo_path = Column(String, nullable=True)

    attendances = relationship('Attendance', back_populates='student', cascade='all,delete')


class Attendance(Base):
    __tablename__ = 'attendances'
    id = Column(Integer, primary_key=True)
    student_id = Column(Integer, ForeignKey('students.id'), nullable=False)
    date = Column(Date, default=date.today, nullable=False)
    time = Column(DateTime, default=datetime.utcnow, nullable=False)
    status = Column(String, default='Puntual')  # Puntual | Tarde | Ausente

    student = relationship('Student', back_populates='attendances')
    __table_args__ = (
        UniqueConstraint('student_id', 'date', name='uq_attendance_student_date'),
    )


def init_db():
    # Crear tablas
    Base.metadata.create_all(ENGINE)

    # Migración ligera para agregar columnas si no existen (SQLite)
    with ENGINE.connect() as conn:
        res = conn.exec_driver_sql("PRAGMA table_info('students')").fetchall()
        existing_cols = {r[1] for r in res}
        if 'registration_date' not in existing_cols:
            conn.exec_driver_sql("ALTER TABLE students ADD COLUMN registration_date DATE")
        if 'photo_path' not in existing_cols:
            conn.exec_driver_sql("ALTER TABLE students ADD COLUMN photo_path VARCHAR")
